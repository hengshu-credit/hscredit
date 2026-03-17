hscredit导入的写法sys.path.insert(0, '/Users/xiaoxi/CodeBuddy/hscredit/hscredit')
置入的规则：青云24 < 650 & 青云24 >= 550
置出的规则：百融定制分V9 < 680 | 青云24 < 550
用于坏账预估的评分：中智小牛分C3
导入包的写法使用sys.path.insert(0, '/Users/xiaoxi/CodeBuddy/hscredit/hscredit')
针对金融信贷场景中的评分，通常值域非[0, 1]的都是分数越大风险越低，而值域在[0, 1]的都是分数越小风险越低。
