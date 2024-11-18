

# TODO
- [ ] input_ids 用的是list，后续需要改成tensor
- [ ] data组合放在model里面不太合理
- [x] stage 训练配置更多放在config里
- [ ] 合并一下重复代码
- [ ] Audio token mask 确认
- [ ] 整理log
- [ ] 支持多卡



# 一些问题

## text

text是target和text是input的时候token的组成不同，需要分开处理。

## audio

audio token mask 需要确认。