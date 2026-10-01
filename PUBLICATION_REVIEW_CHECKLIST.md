# 发布前逐项确认（该 ZIP 是审核稿，不代表已获公开授权）

- [ ] 已得到量霸科技对代码、策略结构、研究结果/文档可公开范围的书面授权；若有限制，进一步裁剪 `src/`, `config/settings.yaml`, `docs/RESEARCH_NOTES.md`。
- [ ] 确认第三方数据的再分发许可。展示包已剔除所有历史行情、新闻文本、LLM 缓存及完整 outputs；如果将来添加图表，确认其来源与披露权限。
- [ ] 原 ZIP 中的 `config/openai_config.py`、`config/tushare_token.py` 有非空且非明显示例值；审核是否是真实有效凭据，如是则更换/撤销。严禁上传原始 ZIP / 将其放入 release。
- [ ] 本包已替换成仅从环境变量读取的安全配置，但请人工再扫描其余脚本、README、PowerShell、仓库历史。
- [ ] 确认公开 README 与真实实验结果一致；不要把代码就绪/默认 dry-run 误写成实盘接入成功。
- [ ] 在干净环境运行 `python generate_demo_data.py`、`python run_pipeline.py`，并运行 `python -m pytest -q`；对于依赖专有数据的测试需单独标记。
- [ ] 测试 V7.8 / V7.9 时自行获取合规数据，不能把完整实习数据夹重新上传。
- [ ] 根据代码权属选择 LICENSE；目前没有 LICENSE 是有意的，不代表任意人可自由重用。
- [ ] 在新目录中 `git init` 并创建第一笔干净 commit；不要带入任何旧的包含秘密的 `.git` 历史。

本审核包只包含源代码和方法论说明。保留了原始项目的核心研究脚本、src/ 和 tests/，不包含原始密钥、实际数据、历史回测输出/LLM 缓存或图片。
