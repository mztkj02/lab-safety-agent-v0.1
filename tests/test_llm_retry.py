"""模型调用重试逻辑的测试。

这段逻辑直接决定「80 张图能不能一次跑完」，而免费模型（glm-4v-flash）
并发只有 1~2，429 是常态而非异常。所以行为必须被测试钉死：

- 429 / 5xx 要退避重试，且遵守服务端 Retry-After
- 401/403 鉴权失败要立刻报错（重试只会浪费时间）
- 400（不支持 response_format）要降级重试一次，而不是当失败
- 调用次数只统计真正成功的请求，否则「调用成本」这个指标是假的
"""

import io
import urllib.error

import pytest

from llm import _backoff_seconds, _retry_after_seconds, OpenAICompatibleVisionLLM

MESSAGES = [{"role": "user", "content": "hi"}]
OK_BODY = b'{"choices": [{"message": {"content": "{}"}}]}'


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = {"Retry-After": retry_after} if retry_after else {}
    return urllib.error.HTTPError(
        "https://example.invalid", code, f"err {code}", headers, io.BytesIO(b'{"error":"x"}')
    )


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def llm(monkeypatch):
    monkeypatch.setenv("SAFETY_LLM_API_KEY", "test.key")
    client = OpenAICompatibleVisionLLM(api_key="test.key", base_url="https://example.invalid",
                                       model="glm-4v-flash")
    client.calls = 0
    return client


@pytest.fixture
def no_sleep(monkeypatch):
    """记录退避时长但不真的睡，否则测试要跑几分钟。"""
    slept: list[float] = []
    monkeypatch.setattr("llm.time.sleep", lambda s: slept.append(s))
    return slept


def test_retries_429_then_succeeds(llm, no_sleep, monkeypatch):
    errors = [_http_error(429), _http_error(429)]
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if errors:
            raise errors.pop(0)
        return _FakeResponse(OK_BODY)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = llm._post(MESSAGES)

    assert result["choices"][0]["message"]["content"] == "{}"
    assert calls["n"] == 3, "应重试两次后成功"
    assert llm.calls == 1, "只应统计成功的调用，失败的 429 不应计入成本"
    assert len(no_sleep) == 2, "每次重试前都应退避等待"
    assert no_sleep[1] > no_sleep[0], "退避应递增"


def test_honors_retry_after_header(llm, no_sleep, monkeypatch):
    errors = [_http_error(429, retry_after="7")]

    def fake_urlopen(req, timeout=None):
        if errors:
            raise errors.pop(0)
        return _FakeResponse(OK_BODY)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm._post(MESSAGES)
    assert no_sleep == [7.0], "服务端给了 Retry-After 就该听它的"


def test_auth_error_fails_fast(llm, no_sleep, monkeypatch):
    """401 重试一百次也没用，必须立刻失败并给出可操作的提示。"""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise _http_error(401)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="鉴权失败"):
        llm._post(MESSAGES)
    assert calls["n"] == 1, "鉴权失败不应重试"
    assert no_sleep == []


def test_non_retryable_4xx_fails_fast(llm, no_sleep, monkeypatch):
    """模型名写错之类的 404，重试只会浪费额度。"""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise _http_error(404)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="不可重试"):
        llm._post(MESSAGES)
    assert calls["n"] == 1


def test_400_downgrades_response_format(llm, no_sleep, monkeypatch):
    """部分模型不支持 response_format，应去掉该字段重试而不是直接失败。"""
    seen = []

    def fake_urlopen(req, timeout=None):
        import json
        payload = json.loads(req.data.decode("utf-8"))
        seen.append("response_format" in payload)
        if "response_format" in payload:
            raise _http_error(400)
        return _FakeResponse(OK_BODY)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = llm._post(MESSAGES)

    assert result["choices"][0]["message"]["content"] == "{}"
    assert seen == [True, False], "第二次请求应已去掉 response_format"
    assert llm.calls == 1


def test_gives_up_after_max_attempts(llm, no_sleep, monkeypatch):
    """一直 429 时要有终止条件，并给出「把 --sleep 调大」这种可操作提示。"""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise _http_error(429)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="429"):
        llm._post(MESSAGES)
    assert calls["n"] == 1 + llm.max_retries + llm.rate_limit_retries
    assert llm.calls == 0, "全部失败时不应计入成功调用成本"


def test_backoff_is_capped_and_jittered():
    # base=2.0，抖动系数 0.5~1.0，所以第 1 次退避落在 1~2 秒
    for _ in range(20):
        assert 1.0 <= _backoff_seconds(1) <= 2.0
    for _ in range(20):
        assert _backoff_seconds(10) <= 60.0, "退避不应超过上限"
    # 抖动：同样的 attempt 多次取值不应完全相同
    assert len({round(_backoff_seconds(3), 6) for _ in range(10)}) > 1


def test_retry_after_parsing():
    assert _retry_after_seconds(_http_error(429, retry_after="12")) == 12.0
    assert _retry_after_seconds(_http_error(429)) == 0.0
    assert _retry_after_seconds(_http_error(429, retry_after="以秒计")) == 0.0
    assert _retry_after_seconds(_http_error(429, retry_after="9999")) == 120.0, "过长的等待要截断"
