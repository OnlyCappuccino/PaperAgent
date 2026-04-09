import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logging import setup_logging
from app.workflow.engine import ResearchWorkflow


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', required=True, help='用户提问')
    args = parser.parse_args()

    setup_logging()
    workflow = ResearchWorkflow()
    state = workflow.run(args.query)

    print('\n===== 最终答案 =====\n')
    print(state.draft_answer)
    print('\n===== Critic 结果 =====\n')
    print(state.critique.model_dump() if state.critique else None)
