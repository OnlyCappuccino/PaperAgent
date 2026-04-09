from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logging import setup_logging
from app.workflow.indexing import build_index


if __name__ == '__main__':
    setup_logging()
    # 建立论文内容索引（在 data/papers 目录下）
    count = build_index()
    print(f'索引完成，共写入 {count} 个 chunks。')
