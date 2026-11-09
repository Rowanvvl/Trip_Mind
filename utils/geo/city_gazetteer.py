"""城市与景点词典。

作用有三个，缺一不可：
1. **城市归一**：「京城」「魔都」「山城」→ 标准城市名，检索过滤才有意义；
2. **同名消歧**：西湖在杭州、南山在三亚与重庆，靠城市上下文 + 别名索引区分；
3. **片区归属**：行程聚类要知道「亚龙湾」和「天涯海角」是不是一个片区。

数据可信度声明：
- 城市名、行政区、景点别名：人工维护，可信；
- **经纬度是近似值（GCJ-02，精度约 0.01°），只用于「地图接口不可用时」的距离估算**，
  真实距离一律以高德 `distance` / `direction` 返回为准；
- 门票区间是估算值，仅用于预算测算的兜底，命中语料时优先用语料值。

维护方式：新增城市只需在 `CITY_DATA` 里加一条，索引会自动重建。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

# (景点名, 行政区, 经度, 纬度, 是否室内, 门票下限, 门票上限, 别名（斜杠分隔）)
AttractionRow = Tuple[str, str, float, float, bool, int, int, str]

CITY_DATA: Dict[str, dict] = {
    "三亚": {
        "province": "海南",
        "aliases": ["鹿城", "sanya"],
        "districts": ["吉阳区", "天涯区", "海棠区", "崖州区"],
        "attractions": [
            ("亚龙湾", "吉阳区", 109.6400, 18.2200, False, 0, 0, "亚龙湾沙滩"),
            ("天涯海角", "天涯区", 109.3600, 18.2960, False, 68, 81, "天涯海角游览区"),
            ("蜈支洲岛", "海棠区", 109.7650, 18.3160, False, 128, 168, "蜈支洲"),
            ("南山文化旅游区", "崖州区", 109.2100, 18.3260, False, 108, 129, "南山寺/南山"),
            ("大小洞天", "崖州区", 109.1600, 18.3300, False, 88, 108, ""),
            ("大东海", "吉阳区", 109.5200, 18.2200, False, 0, 0, ""),
            ("三亚湾", "天涯区", 109.4700, 18.2700, False, 0, 0, "椰梦长廊"),
            ("鹿回头", "吉阳区", 109.5000, 18.2100, False, 25, 42, "鹿回头公园"),
            ("西岛", "天涯区", 109.4100, 18.2400, False, 98, 128, "西玳瑁洲"),
            ("崖州古城", "崖州区", 109.1700, 18.3500, False, 0, 30, ""),
        ],
    },
    "成都": {
        "province": "四川",
        "aliases": ["蓉城", "锦城", "chengdu"],
        "districts": ["锦江区", "青羊区", "武侯区", "成华区", "金牛区", "都江堰市"],
        "attractions": [
            ("宽窄巷子", "青羊区", 104.0570, 30.6700, False, 0, 0, "宽巷子/窄巷子"),
            ("锦里", "武侯区", 104.0480, 30.6420, False, 0, 0, "锦里古街"),
            ("武侯祠", "武侯区", 104.0480, 30.6440, False, 50, 60, ""),
            ("杜甫草堂", "青羊区", 104.0290, 30.6600, False, 50, 60, ""),
            ("大熊猫繁育研究基地", "成华区", 104.1450, 30.7350, False, 55, 55, "熊猫基地/大熊猫基地"),
            ("春熙路", "锦江区", 104.0810, 30.6570, False, 0, 0, ""),
            ("太古里", "锦江区", 104.0830, 30.6540, False, 0, 0, "远洋太古里"),
            ("金沙遗址博物馆", "青羊区", 104.0150, 30.6800, True, 60, 70, "金沙遗址"),
            ("人民公园", "青羊区", 104.0620, 30.6570, False, 0, 0, "少城公园"),
            ("都江堰", "都江堰市", 103.6200, 31.0000, False, 80, 90, "都江堰景区"),
            ("青城山", "都江堰市", 103.5700, 30.9000, False, 80, 90, "青城前山"),
        ],
    },
    "云南": {
        "province": "云南",
        "aliases": ["云南省", "滇", "yunnan"],
        "districts": ["昆明", "大理", "丽江", "香格里拉", "西双版纳"],
        "region_children": ["昆明", "大理", "丽江"],
        "attractions": [
            ("滇池", "昆明", 102.6700, 24.8100, False, 0, 0, "昆明滇池/海埂大坝"),
            ("石林", "昆明", 103.3200, 24.8100, False, 130, 175, "石林风景区"),
            ("翠湖公园", "昆明", 102.7100, 25.0500, False, 0, 0, "翠湖"),
            ("大理古城", "大理", 100.1600, 25.7000, False, 0, 0, ""),
            ("洱海", "大理", 100.1900, 25.7500, False, 0, 0, "环洱海"),
            ("喜洲古镇", "大理", 100.1300, 25.8500, False, 0, 30, "喜洲"),
            ("双廊", "大理", 100.1900, 25.9000, False, 0, 0, "双廊古镇"),
            ("丽江古城", "丽江", 100.2300, 26.8700, False, 50, 50, "大研古城"),
            ("玉龙雪山", "丽江", 100.1800, 27.1000, False, 100, 240, "玉龙雪山景区"),
            ("束河古镇", "丽江", 100.2000, 26.9100, False, 0, 0, "束河"),
        ],
    },
    "杭州": {
        "province": "浙江",
        "aliases": ["杭", "临安", "hangzhou"],
        "districts": ["西湖区", "上城区", "拱墅区", "滨江区", "余杭区", "淳安县"],
        "attractions": [
            ("西湖", "西湖区", 120.1500, 30.2500, False, 0, 0, "西湖景区/断桥"),
            ("灵隐寺", "西湖区", 120.1000, 30.2400, False, 45, 75, "灵隐"),
            ("西溪湿地", "西湖区", 120.0700, 30.2700, False, 80, 100, "西溪"),
            ("河坊街", "上城区", 120.1700, 30.2400, False, 0, 0, "清河坊"),
            ("千岛湖", "淳安县", 119.0400, 29.6000, False, 120, 180, "千岛湖景区"),
            ("宋城", "西湖区", 120.1000, 30.1700, False, 300, 380, "宋城千古情"),
            ("雷峰塔", "西湖区", 120.1500, 30.2300, False, 40, 40, ""),
            ("龙井村", "西湖区", 120.1100, 30.2200, False, 0, 0, "龙井茶园"),
            ("良渚古城遗址", "余杭区", 120.0000, 30.3900, False, 60, 80, "良渚"),
        ],
    },
    "厦门": {
        "province": "福建",
        "aliases": ["鹭岛", "xiamen"],
        "districts": ["思明区", "湖里区", "集美区", "海沧区", "同安区"],
        "attractions": [
            ("鼓浪屿", "思明区", 118.0700, 24.4500, False, 35, 100, "鼓浪屿岛"),
            ("曾厝垵", "思明区", 118.1200, 24.4300, False, 0, 0, "曾厝安"),
            ("厦门大学", "思明区", 118.0900, 24.4400, False, 0, 0, "厦大"),
            ("南普陀寺", "思明区", 118.0900, 24.4450, False, 0, 0, "南普陀"),
            ("环岛路", "思明区", 118.1300, 24.4300, False, 0, 0, "环岛路骑行"),
            ("集美学村", "集美区", 118.1000, 24.5700, False, 0, 0, "集美"),
            ("中山路", "思明区", 118.0800, 24.4600, False, 0, 0, "中山路步行街"),
            ("胡里山炮台", "思明区", 118.1000, 24.4350, False, 25, 25, ""),
            ("园林植物园", "思明区", 118.1100, 24.4500, False, 30, 40, "万石植物园"),
        ],
    },
    "张家界": {
        "province": "湖南",
        "aliases": ["大庸", "zhangjiajie"],
        "districts": ["永定区", "武陵源区", "慈利县", "桑植县"],
        "attractions": [
            ("张家界国家森林公园", "武陵源区", 110.4400, 29.3200, False, 225, 228, "森林公园/武陵源核心景区"),
            ("袁家界", "武陵源区", 110.4300, 29.3400, False, 0, 0, "阿凡达取景地"),
            ("天子山", "武陵源区", 110.4200, 29.3800, False, 0, 0, ""),
            ("金鞭溪", "武陵源区", 110.4500, 29.3300, False, 0, 0, ""),
            ("天门山", "永定区", 110.4800, 29.0500, False, 235, 278, "天门山索道"),
            ("大峡谷玻璃桥", "慈利县", 110.7000, 29.3000, False, 130, 190, "玻璃桥/张家界大峡谷"),
            ("黄龙洞", "慈利县", 110.6000, 29.3600, True, 100, 100, ""),
            ("宝峰湖", "武陵源区", 110.4700, 29.3300, False, 96, 96, ""),
            ("凤凰古城", "湘西州", 109.6000, 27.9500, False, 0, 148, "凤凰"),
        ],
    },
    "重庆": {
        "province": "重庆",
        "aliases": ["山城", "渝", "chongqing"],
        "districts": ["渝中区", "南岸区", "沙坪坝区", "江北区", "武隆区", "大足区"],
        "attractions": [
            ("洪崖洞", "渝中区", 106.5800, 29.5600, False, 0, 0, ""),
            ("解放碑", "渝中区", 106.5800, 29.5570, False, 0, 0, "解放碑步行街"),
            ("磁器口", "沙坪坝区", 106.4500, 29.5800, False, 0, 0, "磁器口古镇"),
            ("长江索道", "渝中区", 106.5800, 29.5500, False, 20, 30, "索道"),
            ("李子坝", "渝中区", 106.5300, 29.5500, False, 0, 0, "李子坝轻轨站"),
            ("武隆天生三桥", "武隆区", 107.7900, 29.4700, False, 95, 135, "天生三桥/武隆"),
            ("大足石刻", "大足区", 105.8000, 29.7000, False, 115, 135, "大足"),
            ("南山一棵树", "南岸区", 106.5800, 29.5300, False, 20, 30, "一棵树观景台"),
            ("鹅岭二厂", "渝中区", 106.5400, 29.5500, False, 0, 0, "二厂文创公园"),
            ("朝天门", "渝中区", 106.5900, 29.5700, False, 0, 0, "来福士"),
        ],
    },
    "北京": {
        "province": "北京",
        "aliases": ["京城", "帝都", "beijing", "京"],
        "districts": ["东城区", "西城区", "朝阳区", "海淀区", "延庆区", "怀柔区"],
        "attractions": [
            ("故宫", "东城区", 116.3970, 39.9160, False, 40, 60, "故宫博物院/紫禁城"),
            ("天安门广场", "东城区", 116.3970, 39.9050, False, 0, 0, "天安门"),
            ("八达岭长城", "延庆区", 116.0200, 40.3600, False, 35, 40, "八达岭"),
            ("慕田峪长城", "怀柔区", 116.5700, 40.4300, False, 45, 60, "慕田峪"),
            ("颐和园", "海淀区", 116.2700, 39.9900, False, 20, 60, ""),
            ("天坛", "东城区", 116.4100, 39.8800, False, 15, 34, "天坛公园"),
            ("圆明园", "海淀区", 116.3000, 40.0000, False, 10, 25, ""),
            ("什刹海", "西城区", 116.3900, 39.9400, False, 0, 0, "后海"),
            ("南锣鼓巷", "东城区", 116.4000, 39.9400, False, 0, 0, ""),
            ("雍和宫", "东城区", 116.4200, 39.9500, False, 25, 25, ""),
            ("国家博物馆", "东城区", 116.4000, 39.9050, True, 0, 0, "国博/中国国家博物馆"),
            ("798艺术区", "朝阳区", 116.4900, 39.9800, False, 0, 0, "798"),
        ],
    },
    "上海": {
        "province": "上海",
        "aliases": ["魔都", "沪", "shanghai", "申城"],
        "districts": ["黄浦区", "徐汇区", "静安区", "浦东新区", "青浦区", "长宁区"],
        "attractions": [
            ("外滩", "黄浦区", 121.4900, 31.2400, False, 0, 0, "外滩万国建筑"),
            ("豫园", "黄浦区", 121.4900, 31.2300, False, 30, 40, "城隍庙"),
            ("南京路步行街", "黄浦区", 121.4800, 31.2350, False, 0, 0, "南京路"),
            ("上海迪士尼", "浦东新区", 121.6700, 31.1400, False, 475, 799, "迪士尼乐园"),
            ("陆家嘴", "浦东新区", 121.5000, 31.2400, False, 0, 180, "东方明珠/上海中心"),
            ("武康路", "徐汇区", 121.4400, 31.2100, False, 0, 0, "武康大楼"),
            ("田子坊", "黄浦区", 121.4700, 31.2100, False, 0, 0, ""),
            ("上海博物馆", "黄浦区", 121.4700, 31.2300, True, 0, 0, "上博"),
            ("朱家角", "青浦区", 121.0500, 31.1100, False, 0, 60, "朱家角古镇"),
            ("静安寺", "静安区", 121.4500, 31.2250, False, 50, 50, ""),
        ],
    },
    "西安": {
        "province": "陕西",
        "aliases": ["长安", "古城", "xian", "镐京"],
        "districts": ["雁塔区", "碑林区", "莲湖区", "新城区", "临潼区", "渭南"],
        "attractions": [
            ("兵马俑", "临潼区", 109.2800, 34.3800, False, 120, 120, "秦始皇兵马俑/秦陵"),
            ("华清宫", "临潼区", 109.2100, 34.3600, False, 120, 150, "华清池"),
            ("大雁塔", "雁塔区", 108.9600, 34.2200, False, 40, 50, "大慈恩寺"),
            ("大唐不夜城", "雁塔区", 108.9600, 34.2100, False, 0, 0, "不夜城"),
            ("西安城墙", "碑林区", 108.9400, 34.2600, False, 54, 54, "明城墙"),
            ("陕西历史博物馆", "雁塔区", 108.9500, 34.2250, True, 0, 30, "陕历博/陕博"),
            ("回民街", "莲湖区", 108.9400, 34.2650, False, 0, 0, "回坊"),
            ("钟楼", "碑林区", 108.9400, 34.2600, False, 30, 50, "钟鼓楼"),
            ("鼓楼", "莲湖区", 108.9400, 34.2620, False, 30, 50, ""),
            ("华山", "渭南", 110.0800, 34.4800, False, 100, 160, "西岳华山"),
        ],
    },
    "广州": {
        "province": "广东",
        "aliases": ["羊城", "穗", "花城", "guangzhou"],
        "districts": ["越秀区", "荔湾区", "海珠区", "天河区", "白云区", "番禺区"],
        "attractions": [
            ("广州塔", "海珠区", 113.3200, 23.1050, False, 150, 228, "小蛮腰"),
            ("沙面", "荔湾区", 113.2400, 23.1050, False, 0, 0, "沙面岛"),
            ("陈家祠", "荔湾区", 113.2500, 23.1250, True, 10, 10, "陈氏书院"),
            ("上下九", "荔湾区", 113.2400, 23.1150, False, 0, 0, "上下九步行街"),
            ("北京路", "越秀区", 113.2700, 23.1250, False, 0, 0, "北京路步行街"),
            ("白云山", "白云区", 113.2900, 23.1700, False, 5, 20, ""),
            ("长隆旅游度假区", "番禺区", 113.3200, 23.0000, False, 250, 350, "长隆"),
            ("石室圣心大教堂", "越秀区", 113.2600, 23.1150, True, 0, 0, "石室"),
            ("永庆坊", "荔湾区", 113.2400, 23.1100, False, 0, 0, ""),
            ("越秀公园", "越秀区", 113.2600, 23.1400, False, 0, 0, "五羊石像"),
        ],
    },
    "南京": {
        "province": "江苏",
        "aliases": ["金陵", "宁", "石头城", "nanjing"],
        "districts": ["玄武区", "秦淮区", "建邺区", "鼓楼区", "栖霞区", "江宁区"],
        "attractions": [
            ("中山陵", "玄武区", 118.8500, 32.0600, False, 0, 0, "中山陵景区"),
            ("明孝陵", "玄武区", 118.8300, 32.0600, False, 70, 70, ""),
            ("夫子庙", "秦淮区", 118.7900, 32.0200, False, 0, 30, "夫子庙秦淮风光带"),
            ("秦淮河", "秦淮区", 118.7900, 32.0200, False, 0, 100, "夜游秦淮"),
            ("总统府", "玄武区", 118.7900, 32.0450, True, 40, 40, "南京总统府"),
            ("南京博物院", "玄武区", 118.8300, 32.0400, True, 0, 0, "南博"),
            ("侵华日军南京大屠杀遇难同胞纪念馆", "建邺区", 118.7300, 32.0350, True, 0, 0, "大屠杀纪念馆"),
            ("玄武湖", "玄武区", 118.8000, 32.0700, False, 0, 0, ""),
            ("老门东", "秦淮区", 118.7900, 32.0150, False, 0, 0, ""),
            ("栖霞山", "栖霞区", 118.9600, 32.1500, False, 25, 40, "栖霞寺"),
        ],
    },
    "武汉": {
        "province": "湖北",
        "aliases": ["江城", "汉口", "wuhan"],
        "districts": ["武昌区", "江汉区", "汉阳区", "洪山区", "江岸区"],
        "attractions": [
            ("黄鹤楼", "武昌区", 114.3000, 30.5450, False, 70, 70, ""),
            ("东湖", "武昌区", 114.4000, 30.5500, False, 0, 60, "东湖绿道"),
            ("户部巷", "武昌区", 114.3000, 30.5500, False, 0, 0, ""),
            ("昙华林", "武昌区", 114.3100, 30.5600, False, 0, 0, ""),
            ("武汉大学", "武昌区", 114.3600, 30.5350, False, 0, 0, "武大/珞珈山"),
            ("湖北省博物馆", "武昌区", 114.3700, 30.5600, True, 0, 0, "省博"),
            ("江汉路", "江汉区", 114.2900, 30.5800, False, 0, 0, "江汉路步行街"),
            ("武汉长江大桥", "汉阳区", 114.2900, 30.5500, False, 0, 0, "长江大桥"),
            ("归元寺", "汉阳区", 114.2500, 30.5500, False, 10, 10, ""),
            ("光谷", "洪山区", 114.4000, 30.4800, False, 0, 0, "光谷广场"),
        ],
    },
    "长沙": {
        "province": "湖南",
        "aliases": ["星城", "潭州", "changsha"],
        "districts": ["岳麓区", "天心区", "开福区", "芙蓉区", "雨花区", "望城区"],
        "attractions": [
            ("岳麓山", "岳麓区", 112.9400, 28.1800, False, 0, 0, "岳麓山景区/爱晚亭"),
            ("橘子洲", "岳麓区", 112.9600, 28.1900, False, 0, 0, "橘子洲头"),
            ("湖南省博物馆", "开福区", 112.9900, 28.2100, True, 0, 0, "省博/马王堆"),
            ("太平街", "天心区", 112.9700, 28.1900, False, 0, 0, "太平老街"),
            ("坡子街", "天心区", 112.9700, 28.1850, False, 0, 0, ""),
            ("文和友", "天心区", 112.9800, 28.1800, False, 0, 0, "超级文和友"),
            ("天心阁", "天心区", 112.9800, 28.1800, False, 32, 32, ""),
            ("铜官窑", "望城区", 112.8000, 28.4500, False, 100, 198, "铜官窑古镇"),
            ("梅溪湖", "岳麓区", 112.9000, 28.1900, False, 0, 0, "梅溪湖艺术中心"),
            ("世界之窗", "开福区", 113.0400, 28.2300, False, 180, 260, "长沙世界之窗"),
        ],
    },
}


@dataclass
class Attraction:
    name: str
    city: str
    district: str
    lng: float
    lat: float
    indoor: bool
    ticket_low: int
    ticket_high: int
    aliases: List[str] = field(default_factory=list)

    @property
    def all_names(self) -> List[str]:
        return [self.name, *self.aliases]

    @property
    def ticket_avg(self) -> int:
        return int(round((self.ticket_low + self.ticket_high) / 2))

    @property
    def free(self) -> bool:
        return self.ticket_high == 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "city": self.city,
            "district": self.district,
            "lng": self.lng,
            "lat": self.lat,
            "indoor": self.indoor,
            "ticket_low": self.ticket_low,
            "ticket_high": self.ticket_high,
        }


def _build_indexes():
    city_alias: Dict[str, str] = {}
    attraction_index: Dict[str, List[Attraction]] = {}
    attractions_by_city: Dict[str, List[Attraction]] = {}

    for city, info in CITY_DATA.items():
        city_alias[city] = city
        for alias in info.get("aliases", []):
            city_alias[alias.lower()] = city
        for child in info.get("region_children", []):
            # 子城市（昆明/大理/丽江）也登记为可识别城市，但归属到省级区域
            city_alias[child] = child
        bucket: List[Attraction] = []
        for row in info.get("attractions", []):
            name, district, lng, lat, indoor, low, high, alias_str = row
            aliases = [a for a in alias_str.split("/") if a]
            item = Attraction(
                name=name,
                city=city,
                district=district,
                lng=lng,
                lat=lat,
                indoor=indoor,
                ticket_low=low,
                ticket_high=high,
                aliases=aliases,
            )
            bucket.append(item)
            for token in item.all_names:
                attraction_index.setdefault(token, []).append(item)
        attractions_by_city[city] = bucket
    return city_alias, attraction_index, attractions_by_city


CITY_ALIAS_INDEX, ATTRACTION_INDEX, ATTRACTIONS_BY_CITY = _build_indexes()

# 省级区域 → 子城市（用于过滤放宽）
REGION_CHILDREN: Dict[str, List[str]] = {
    city: info.get("region_children", []) for city, info in CITY_DATA.items() if info.get("region_children")
}

ALL_CITIES: List[str] = list(CITY_DATA.keys())


_CITY_SUFFIX = re.compile(r"(多城市|多城|多地|全域|全省|全州|全境|地区|周边|一带)$")
# 长词优先：避免「云南大理」被短别名抢先匹配
_ALIAS_BY_LEN: List[Tuple[str, str]] = sorted(CITY_ALIAS_INDEX.items(), key=lambda kv: len(kv[0]), reverse=True)


def normalize_city(city: str | None) -> str:
    """把别名归一成标准名；无法识别时返回去空白后的原文（由调用方决定是否报错）。

    语料里的城市字段写法很杂：`云南多城市`、`北京市`、`云南大理`。
    前两种要归到标准名，否则 `city` 过滤会静默漏召——
    存成「云南多城」的切片，用户问「云南」时一条都匹配不上。
    """
    if not city:
        return ""
    text = str(city).strip().lower().replace("市", "").replace("省", "")
    if text in CITY_ALIAS_INDEX:
        return CITY_ALIAS_INDEX[text]
    # 「云南多城市」「上海周边」这类区域描述，去掉尾巴再查
    trimmed = _CITY_SUFFIX.sub("", text)
    if trimmed and trimmed in CITY_ALIAS_INDEX:
        return CITY_ALIAS_INDEX[trimmed]
    # 「云南大理」这类「省+市」写法，取后段
    for alias, standard in _ALIAS_BY_LEN:
        if len(alias) >= 2 and text.endswith(alias):
            return standard
    # 「北京朝阳区」这类「市+区」写法，取前段
    for alias, standard in _ALIAS_BY_LEN:
        if len(alias) >= 2 and text.startswith(alias):
            return standard
    return trimmed or text


def is_known_city(city: str | None) -> bool:
    normalized = normalize_city(city)
    return normalized in CITY_DATA or normalized in {c for cs in REGION_CHILDREN.values() for c in cs}


def get_city_info(city: str) -> dict:
    standard = normalize_city(city)
    if standard in CITY_DATA:
        return CITY_DATA[standard]
    for parent, children in REGION_CHILDREN.items():
        if standard in children:
            return CITY_DATA[parent]
    return {}


def city_scope(city: str | None) -> dict:
    """检索过滤的取值域。

    区域类目的地（云南）需要同时匹配「云南」和其子城市（昆明/大理/丽江）的切片，
    否则「云南行程」会漏召回一半文档。
    """
    standard = normalize_city(city) if city else ""
    if not standard:
        return {"city": None, "cities": [], "province": None}
    children = REGION_CHILDREN.get(standard, [])
    return {
        "city": standard,
        "cities": [standard, *children] if children else [standard],
        "province": get_city_info(standard).get("province"),
    }


def find_city_in_text(text: str) -> Optional[str]:
    """从自由文本里找城市，长词优先（避免「南京路」被判成南京）。"""
    if not text:
        return None
    lowered = text.lower()
    candidates = sorted(CITY_ALIAS_INDEX.keys(), key=len, reverse=True)
    for alias in candidates:
        if len(alias) < 2:
            continue
        if alias in lowered:
            standard = CITY_ALIAS_INDEX[alias]
            # 「南京路」「北京路」是步行街，不是城市
            if f"{alias}路" in lowered and standard in {"南京", "北京"}:
                continue
            return standard
    return None


def find_cities_in_text(text: str) -> List[str]:
    """多城市识别（「北京上海 7 天」），按出现顺序返回。"""
    if not text:
        return []
    hits: List[Tuple[int, str]] = []
    lowered = text.lower()
    for alias, standard in CITY_ALIAS_INDEX.items():
        if len(alias) < 2:
            continue
        pos = lowered.find(alias)
        if pos >= 0:
            hits.append((pos, standard))
    hits.sort()
    ordered: List[str] = []
    for _, standard in hits:
        if standard not in ordered:
            ordered.append(standard)
    return ordered


def resolve_attraction(name: str, city: str | None = None) -> Optional[Attraction]:
    """景点消歧：有城市上下文时优先该城市，否则返回最可能的那个。"""
    if not name:
        return None
    key = name.strip()

    # 有城市上下文时，先在该城市的景点里做包含匹配——
    # 「南山」在三亚是南山文化旅游区，在重庆是南山一棵树，只靠全局索引会串城。
    if city:
        for item in attractions_of(city):
            for token in item.all_names:
                if token == key or key in token or token in key:
                    return item

    items = ATTRACTION_INDEX.get(key)
    if not items:
        # 模糊：别名包含
        for token, bucket in ATTRACTION_INDEX.items():
            if token and token in key:
                items = bucket
                break
    if not items:
        return None
    return items[0]


def district_of(attraction_name: str, city: str | None = None) -> Optional[str]:
    item = resolve_attraction(attraction_name, city)
    return item.district if item else None


def attractions_of(city: str) -> List[Attraction]:
    standard = normalize_city(city)
    if standard in ATTRACTIONS_BY_CITY:
        return ATTRACTIONS_BY_CITY[standard]
    for parent, children in REGION_CHILDREN.items():
        if standard in children:
            return [a for a in ATTRACTIONS_BY_CITY.get(parent, []) if a.district == standard]
    return []


def find_attractions_in_text(text: str, city: str | None = None) -> List[Attraction]:
    """抽出文本里提到的景点（长名优先，避免「南山」吃掉「南山文化旅游区」）。"""
    found: List[Attraction] = []
    seen = set()
    tokens = sorted(ATTRACTION_INDEX.keys(), key=len, reverse=True)
    for token in tokens:
        if len(token) < 2 or token not in text:
            continue
        for item in ATTRACTION_INDEX[token]:
            if city and normalize_city(city) not in (item.city, item.district):
                continue
            if item.name in seen:
                continue
            seen.add(item.name)
            found.append(item)
    return found


def indoor_attractions(city: str) -> List[Attraction]:
    """雨天替换用：室内项目。"""
    return [a for a in attractions_of(city) if a.indoor]


def summarize() -> dict:
    return {
        "cities": len(CITY_DATA),
        "attractions": sum(len(v) for v in ATTRACTIONS_BY_CITY.values()),
        "aliases": len(CITY_ALIAS_INDEX),
    }
