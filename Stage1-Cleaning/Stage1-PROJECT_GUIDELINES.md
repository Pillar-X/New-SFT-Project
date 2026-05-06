你现在是我的项目工程助手。我要开始一个 LLM / 数据清洗 / 微调类项目。请你先帮我搭建一个清晰、可维护、可复现的项目结构，而不是直接把所有代码写进 main.py。

请按以下结构组织项目：

project/
├── README.md
├── requirements.txt
├── .gitignore
├── Makefile
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── samples/
├── models/
├── src/
│   └── project_name/
├── scripts/
├── evaluation/
├── notebooks/
├── tests/
├── outputs/
└── logs/

3. 每个目录的职责如下：  
- src/：只放可复用函数、类和模块，不直接运行完整流程。  
- scripts/：只放命令行入口，每个脚本完成一个明确流程。  
- configs/：放 YAML 配置文件，路径、模型名、batch size、max samples 等参数都放这里。  
- data/raw/：放原始数据，默认不手动修改。  
- data/interim/：放中间处理结果。  
- data/processed/：放最终训练、验证、测试数据。  
- data/samples/：放小样本调试数据。  
- notebooks/：只用于探索、画图、错误分析，不作为正式 pipeline。  
- tests/：放单元测试。  
- outputs/：放实验输出，每次实验应有独立子目录。  
- logs/：放运行日志。
- models/: 放模型
- evaluation/: 和测评相关的结果

4. 代码维护规则：  
- 所有可复用逻辑必须写进 src/。  
- 所有正式运行入口必须写进 scripts/。  
- 所有参数必须尽量写进 configs/，不要硬编码在代码里。  
- 所有运行命令必须写进 README.md。  
- 如果写了临时测试代码，必须放到 notebooks/、tests/ 或 scratch 文件中，不要混入正式脚本。  
- 不要在正式脚本中保留无关的 print、exit、临时路径、临时 debug 逻辑。  
- 模块文件中不要在 import 时自动执行任务。  
- 每个正式脚本必须包含 main() 和 if __name__ == "__main__": main()。

5. 每个 scripts/ 下的脚本都要遵守这个模板：  
  
import argparse  
  
def parse_args():  
parser = argparse.ArgumentParser()  
parser.add_argument("--config", type=str, default="configs/default.yaml")  
return parser.parse_args()  
  
def main():  
args = parse_args()  
# 1. load config  
# 2. run one clear pipeline step  
# 3. save output  
# 4. print key statistics  
  
if __name__ == "__main__":  
main()

6. README.md 必须包含：  
- 项目目标  
- 目录结构说明  
- 环境安装命令  
- 每一步 pipeline 的运行命令  
- 数据流说明  
- 输出文件说明  
- 常见问题和注意事项
README 中至少要包含类似命令：  
  
conda create -n llm_project python=3.11  
conda activate llm_project  
pip install -r requirements.txt  
  
python scripts/00_inspect_data.py --config configs/default.yaml  
python scripts/01_filter_data.py --config configs/default.yaml

6. Makefile 中要提供常用命令，假如scripts中有以下文件，则应该提供命令：  
inspect:  
python scripts/00_inspect_data.py --config configs/default.yaml  
filter:  
python scripts/01_filter_data.py --config configs/default.yaml

7. 配置文件 configs/default.yaml 至少要包含：  
  
project:  
name: project_name  
seed: 42  
  
paths:  
raw_data: data/raw/input.jsonl  
interim_dir: data/interim  
processed_dir: data/processed  
output_dir: outputs  
log_dir: logs  
  
model:  
name: Qwen/Qwen2.5-0.5B-Instruct  
base_url: http://localhost:8000/v1  
api_key: dummy  
temperature: 0.2  
max_tokens: 1024  
  
processing:  
max_samples: null  
batch_size: 16  
  
validation:  
min_question_length: 5  
min_answer_length: 5  
require_final_answer: false  
  
9. 每次实验输出目录应尽量包含：  
- config.yaml：本次实验使用的配置副本  
- command.txt：本次运行命令  
- metrics.json：关键指标  
- samples.jsonl：部分输出样例  
- log.txt：运行日志  
  
10. 如果我让你新增功能，请先判断它应该放在哪里：  
- 可复用逻辑 → src/  
- 命令行流程 → scripts/  
- 参数 → configs/  
- 探索分析 → notebooks/  
- 单元测试 → tests/  
- 输出结果 → outputs/  
- 数据文件 → data/  
  
11. 如果你发现我把临时代码、测试代码、debug 代码混进正式脚本，请主动指出，并建议移到合适位置。  
  
12. 如果你发现某个脚本职责过多，请主动建议拆分。例如：  
- 数据读取和清洗混在一起时，可以把读取放到 io.py，清洗放到 rules.py。  
- 生成 SFT 和验证 SFT 混在一起时，应拆成 generate_sft.py 和 validate_sft.py。  
- prompt 模板不要散落在多个脚本中，应统一放到 prompts.py。  
- 数据格式转换应放到 sft.py。  
- 质量检查规则应放到 validate.py。

13. 你生成代码时，要优先保证：  
- 结构清晰  
- 可以命令行运行  
- 可以小样本调试  
- 可以记录输出  
- 可以后续扩展  
- 不要过度复杂化  

请先初始化项目结构，并生成 README.md、requirements.txt、.gitignore、Makefile、configs/default.yaml，以及 src/ 和 scripts/ 下的基础代码骨架。