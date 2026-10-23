"""把 ImageGen 生成的实验室场景图整理进数据集目录，并生成可追溯的清单。

背景
----
公开图库（Pexels 等）里几乎不存在「违规现场」的照片 —— 没有人会专门拍摄
「试剂瓶没标签」「实验台上放咖啡杯」「地面洒了一滩液体还没处理」这类画面。
所以本项目采用两条素材来源：

  A. 真实网络图片（Pexels License）→ 正常场景 + 易混淆样本   data/web_raw/
  B. 文生图合成（本脚本处理）      → 明确的违规场景         data/synth/

关键约束：合成图必须在清单里标记 source=synthetic，报告中不得与真实拍摄混为一谈。

本脚本做的事：
  1. 按生成时间顺序把 raw 目录里的 PNG 映射到风险类别；
  2. 裁掉底部的水印条、统一转成 JPEG（与 Pexels 素材格式一致）；
  3. 写出 data/synth/synth_manifest.json，记录每张图对应的提示词原文。

用法：
    python scripts/prepare_synth_data.py --src "<生成的图片目录>"
"""

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTH_DIR = PROJECT_ROOT / "data" / "synth"
MANIFEST = SYNTH_DIR / "synth_manifest.json"

#: 按生成时间升序排列的 (类别, 证据描述, 提示词要点)
#: 顺序必须与调用文生图的先后一致 —— 文件名是随机前缀，只能靠时间排序对齐。
PLAN = [
    ("ppe_missing", "学生在台前操作玻璃器皿，只戴普通近视镜，未戴护目镜、未穿实验服",
     "本科生化学实验室，学生穿便装操作玻璃器皿，只戴日常眼镜，无护目镜无实验服"),
    ("ppe_missing", "两名学生在通风橱前操作，均穿便装未戴护目镜，一人手持烧杯",
     "分析化学实验室夜间，两名学生便装、无护目镜，手持盛有清液的烧杯"),
    ("ppe_missing", "俯视台面，学生裸露手臂短袖操作，无手套无护目镜",
     "俯拍有机化学教学实验室台面，学生短袖裸露手臂倾倒液体，无手套无护目镜"),
    ("ppe_missing", "女生穿白大褂但护目镜挂在颈部未佩戴，徒手拿烧杯",
     "生物化学实验室，女生穿白大褂但护目镜挂在脖子上，徒手拿黄色液体烧杯"),
    ("reagent_management", "台面三个试剂瓶标签全空白，其中一个敞口未盖盖",
     "材料实验室台面，三个试剂瓶标签空白，其中一个开盖"),
    ("reagent_management", "试剂架上多个棕色/透明瓶标签被撕掉，无危险象形标识，堆放到架边",
     "旧化学实验室试剂架，多个试剂瓶标签剥落或空白，无危险标识，堆叠到架边"),
    ("reagent_management", "大烧杯盛无色液体敞口无标签，表面皿与盖子丢在一旁",
     "实验台大烧杯盛无色液体，敞口无标签，表面皿和盖子放在一边"),
    ("flammable_near_heat", "电热板通电发红，旁边紧挨一叠活页纸、纸巾和纸箱",
     "实验室电热板发红，旁边紧贴纸张纸巾和纸箱"),
    ("flammable_near_heat", "酒精灯点燃，旁边敞口的乙醇瓶与纸巾距离仅数厘米",
     "点燃的酒精灯旁敞口乙醇瓶与纸巾"),
    ("flammable_near_heat", "加热套工作，旁边塑料量筒与塑料洗瓶紧贴热表面",
     "加热套上的圆底烧瓶，旁边塑料量筒与洗瓶紧贴热表面"),
    ("flammable_near_heat", "通风橱外明火，敞口有机溶剂桶、纸张和抹布在旁",
     "通风橱外本生灯明火，敞口有机溶剂桶与纸张抹布"),
    ("food_or_drink", "实验台上带盖吸管的一次性纸杯，紧邻试管架和烧杯",
     "实验台一次性纸杯带塑料盖吸管，紧邻试管架和烧杯"),
    ("food_or_drink", "台面上透明水瓶与拆封的面包，混在玻璃器皿和移液管之间",
     "凌乱台面上透明水瓶与拆封面包，混在玻璃器皿之间"),
    ("food_or_drink", "不锈钢保温杯放在仪器和笔记本旁",
     "实验台不锈钢保温杯放在仪器和笔记本旁"),
    ("food_or_drink", "珍珠奶茶杯（带粗吸管）立在试管架和烧瓶旁",
     "实验台珍珠奶茶杯带粗吸管，立在试管架和烧瓶旁"),
    ("food_or_drink", "穿白大褂的女生在台边举杯饮用，台上有敞口试剂瓶和电热板",
     "穿白大褂的女生在实验台边喝纸杯咖啡，台上有敞口试剂瓶和电热板"),
    ("spill_or_obstruction", "实验台下方地砖上一滩透明液体，无警示牌无吸附材料",
     "实验室地砖上大面积透明洒漏液体，无警示牌无吸附材料"),
    ("spill_or_obstruction", "实验室过道被纸箱、周转箱和推车完全堵死",
     "实验室过道被纸箱周转箱和推车完全堵塞，无法通行"),
    ("spill_or_obstruction", "棕色液体自台面漫过边缘滴落到地面，玻璃器皿倾倒，无人清理",
     "台面棕色液体溢出并滴到地面，器皿倾倒，无警示"),
    ("spill_or_obstruction", "多根电源线与插排横穿通行走道，椅子压在上面",
     "地面电源线和插排横穿过道，椅子压在上面"),
    ("spill_or_obstruction", "实验台间通道被推车、凳子和仪器箱堵死，安全出口几乎不可见",
     "窄通道被推车凳子和箱子堵死，安全出口不可达"),
    ("spill_or_obstruction", "拖把桶翻倒，地面积水反光，小心地滑警示牌倒在一边",
     "拖把桶翻倒地面积水，警示锥倒在一边"),
    ("spill_or_obstruction", "消防器材柜前堆放纸箱与推车，通道被挤到无法通行",
     "灭火器柜前堆纸箱推车，通道无法通行"),
    ("ppe_missing", "男生穿短袖在敞开的通风橱前用移液管操作，未戴护目镜未穿实验服",
     "男生短袖在通风橱前移液，无护目镜无实验服"),
    ("ppe_missing", "两名学生在台前讨论，一人手持玻璃搅拌棒，均未戴护目镜未穿实验服",
     "两名学生台边讨论，手持搅拌棒，无护目镜无实验服"),
    ("reagent_management", "塑料瓶仅在胶带上留下一片字迹模糊的手写标签，旁边小瓶完全无标签",
     "塑料瓶只有模糊手写标签，旁边小瓶无标签"),
    ("reagent_management", "大塑料溶剂桶敞口缺盖，旁边小烧杯亦敞口无标签",
     "大溶剂桶敞口缺盖，旁边小烧杯敞口无标签"),
    ("reagent_management", "试剂架上小瓶拥挤堆叠，多数无标签或标签背向不可读，未按危险类别分区",
     "试剂架小瓶拥挤堆叠，多数无标签或标签背向，未按危险类别分区"),
    ("flammable_near_heat", "本生灯明火旁放着一本线圈本、多页打印操作规程和纸巾",
     "本生灯明火旁放笔记本、打印纸和纸巾"),
    ("flammable_near_heat", "电热板通电，旁边紧贴塑料酒精喷壶和揉皱的塑料袋，无任何遮挡",
     "电热板通电，旁边紧贴酒精喷壶和塑料袋"),
    ("food_or_drink", "台面上拆开的饼干袋与剥开的香蕉放在滴定架和烧杯旁，台面有碎屑",
     "台面饼干袋与剥开的香蕉放在滴定架旁，有碎屑"),
    # ---- 易混淆样本：画面看起来像违规，实际完全合规 ----
    # 这一类专门用来暴露「只看表面特征就下结论」的误报，是验证证据复核 Agent 的关键。
    # evidence 以【易混淆】开头，方便脚本识别。
    ("normal", "【易混淆】台面上是带印刷标签和弯曲出液嘴的实验室洗瓶，乍看像饮料杯，实为合规器材",
     "带标签的塑料洗瓶，乍看像饮料瓶，实为实验室洗瓶"),
    ("normal", "【易混淆】烧杯中的清水已用表面皿盖好，旁边试剂瓶标签朝外完整可读",
     "烧杯已用表面皿盖好，旁边试剂瓶标签完整朝外"),
    ("normal", "【易混淆】电热板显示屏为 OFF、指示灯不亮、表面冷态，旁边的笔记本不构成热源风险",
     "电热板已断电，显示屏 OFF、指示灯不亮，旁边笔记本无风险"),
    ("normal", "【易混淆】地面干燥，只是窗光形成的长条反光与阴影，容易被误判为液体洒漏",
     "地面干燥，窗光反光与阴影形似洒漏"),
    ("ppe_missing", "【易混淆】本意是生成「已戴口罩但被遮挡」的画面，实际画面中护目镜完全不可见，"
                    "人工按未佩戴标注；这类低可见度样本本身是误判高发区",
     "侧角度、头发遮挡眼部的学生，护目镜实际不可见（生成偏差，按未佩戴标注）"),
    ("normal", "【易混淆】试剂瓶标签朝内背对镜头，但瓶身侧面贴有两个 GHS 危险象形标识，属于规范标识",
     "试剂瓶标签背向，但侧面有 GHS 象形标识，属规范标识"),
    ("normal", "【易混淆】学生侧身站立、护目镜带子清晰可见且正确跨越头部，属规范佩戴（与上一张形成对照）",
     "学生侧身，护目镜带子清晰可见，规范佩戴"),
    ("normal", "【易混淆】烧杯已盖表面皿，后方试剂瓶标签正面朝外、字迹可读，配有洁净药匙",
     "烧杯盖好表面皿，试剂瓶标签朝外可读"),
    ("normal", "【易混淆】电热板已断电且拔掉插头、电源线盘好，面板全暗、表面清洁冷却，规程单平放在安全距离",
     "电热板断电拔插头、线盘好、面板暗，规程单在安全距离"),
    ("normal", "【易混淆】地面完全干燥，仅有一道斜向窗光在亮面瓷砖上形成反光带，砖缝与浮尘清晰可辨",
     "地面干燥，斜向窗光反光带形似湿痕，砖缝清晰"),
    ("normal", "【易混淆】试剂瓶印刷标签背对镜头，但可见侧面贴有菱形 GHS 象形标识与副标签",
     "试剂瓶标签背向，侧面有 GHS 菱形象形与副标签"),
    ("normal", "【易混淆】台面堆放较乱，但所有试剂瓶标签完整可读、瓶盖全部拧紧，无敞口容器",
     "台面杂乱但试剂瓶标签完整、瓶盖全部拧紧"),
    ("ppe_missing", "【易混淆】护目镜被推到额头上而非罩住双眼，学生裸手拿着盛有清液的烧杯操作",
     "护目镜推到额头，裸手拿烧杯操作"),
    ("normal", "【易混淆】走廊角落放有空的塑料桶与折叠凳，均靠墙收纳，通行走道完全畅通",
     "角落空桶与折叠凳靠墙收纳，走道畅通"),
    ("normal", "【易混淆】台面上放着丁腈手套盒与玻璃器皿，略显凌乱，但无敞口化学品、无食品饮料",
     "台面放手套盒与玻璃器皿，略乱但无敞口化学品"),
    ("normal", "【易混淆】通风橱拉门处于正常工作高度，学生在柜内操作并戴好护目镜与手套，导流板与排风栅无遮挡",
     "通风橱拉门高度正常，学生戴护目镜手套柜内操作"),
    ("normal", "【易混淆】广口试剂瓶敞口放置，但标签明确标注为蒸馏水且有配制日期，内容物无危害",
     "广口瓶敞口但标签标注蒸馏水与配制日期"),
    ("normal", "【易混淆】地面有一小滩液体，但已竖立小心地滑警示牌并由戴手套人员用吸附材料清理",
     "地面积水已立警示牌并有人吸附清理"),
]

#: ImageGen 输出统一裁掉底部水印条（原始 1024x1536，水印在右下角）
CROP_BOTTOM = 92


def main() -> None:
    parser = argparse.ArgumentParser(description="整理合成图进数据集")
    parser.add_argument("--src", required=True, help="ImageGen 输出目录")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from PIL import Image

    src = Path(args.src)
    files = sorted(
        [p for p in src.glob("*.png") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
    )
    print(f"源目录 {src} 共 {len(files)} 张 PNG")
    if len(files) < len(PLAN):
        print(f"警告：只找到 {len(files)} 张，PLAN 需要 {len(PLAN)} 张，按较少数处理")
    if len(files) > len(PLAN):
        print(f"警告：多出 {len(files) - len(PLAN)} 张未纳入，请检查 PLAN 是否遗漏")

    counts = {}
    confusable_n = 0
    items = []
    for idx, path in enumerate(files[:len(PLAN)]):
        risk_type, evidence, prompt_brief = PLAN[idx]
        confusable = evidence.startswith("【易混淆】")
        if confusable:
            confusable_n += 1
            name = f"synth_confuse_{confusable_n:02d}"
        else:
            counts[risk_type] = counts.get(risk_type, 0) + 1
            name = f"synth_{risk_type}_{counts[risk_type]:02d}"
        dest = SYNTH_DIR / f"{name}.jpg"
        if args.dry_run:
            print(f"  {path.name}  ->  {dest.name}  [{risk_type}]")
            continue

        SYNTH_DIR.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size
            cropped = rgb.crop((0, 0, w, max(1, h - CROP_BOTTOM)))
            cropped.save(dest, "JPEG", quality=90)

        items.append({
            "sample_id": name,
            "file": f"data/synth/{name}.jpg",
            "risk_type": risk_type,
            "confusable": confusable,
            "evidence": evidence,
            "prompt": prompt_brief,
            "source": "synthetic",
            "generator": "ImageGen (text-to-image)",
            "watermark_removed": True,
            "original": path.name,
        })

    if args.dry_run:
        return

    MANIFEST.write_text(
        json.dumps(
            {
                "note": "本目录全部为文生图合成样本，不是真实拍摄。仅供流程验证与"
                        "「缺真实数据时的占位评测」，报告中不得声明为实拍数据。",
                "crop_bottom_px": CROP_BOTTOM,
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n输出 {SYNTH_DIR}，共 {len(items)} 张")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"清单：{MANIFEST}")

    # 顺带把原始 PNG 备份到同目录，避免继续占用生成的临时目录
    backup = SYNTH_DIR / "_original_png"
    if not backup.exists():
        backup.mkdir(parents=True, exist_ok=True)
        for path in files[:len(PLAN)]:
            shutil.copy2(path, backup / path.name)
        print(f"原始 PNG 已备份到 {backup}")


if __name__ == "__main__":
    main()
