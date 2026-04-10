import argparse

from app.core.logging import setup_logging
from app.workflow.indexing import build_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', required=False, help='启动向量数据库名称(默认为research_chunks)')
    parser.add_argument('--clear', action='store_true', help='是否删除其它collection（慎用）保留当前collection')
    parser.add_argument('--rebuild', action='store_true', help='是否重建collection（会删除当前collection）')
    args = parser.parse_args()
    
    setup_logging()
    # 建立论文内容索引（在data/papers目录下）
    count = build_index(args)
    print(f'索引完成，共写入 {count} 个 chunks。')
