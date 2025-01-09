

# TODO
- [ ] input_ids 用的是list，后续需要改成tensor
- [ ] data组合放在model里面不太合理
- [x] stage 训练配置更多放在config里
- [x] 合并一下重复代码
- [x] Audio token mask 确认
- [ ] 整理log
- [x] 支持多卡



# 一些问题

## text

text是target和text是input的时候token的组成不同，需要分开处理。

## audio

audio token mask 需要确认。


# infer

## generate
### `generate_A1T1`

只通过输入生成第一个，后面的auto-regressive生成。


## next_token

### `next_token_asr`

生成下一个audio tokens
=> 对audio feature进行自回归
与`next_token_A1T2`一样

### `next_token_A1T1`

生成下一个text tokens
=> 对text进行自回归


### 

