"""Byte-Pair Encoding (BPE) task."""

TASK = {
    "title": "字节对编码（BPE）",
    "difficulty": "Hard",
    "function_name": "SimpleBPE",
    "hint": "`train`：将单词拆为字符并追加 `</w>`，反复找到频率最高的相邻符号对并合并。`encode`：按顺序应用学到的合并规则，将文本拆为子词。",
    "tests": [
        {
            "name": "合并次数正确",
            "code": "\nbpe = {fn}()\nbpe.train(['low', 'low', 'low', 'lower', 'newest', 'widest'], num_merges=5)\nassert len(bpe.merges) == 5, f'Expected 5 merges, got {len(bpe.merges)}'\n"
        },
        {
            "name": "优先合并频率最高的符号对",
            "code": "\nbpe = {fn}()\nbpe.train(['aaa', 'aaa', 'aaa', 'bbb'], num_merges=1)\nassert bpe.merges[0] == ('a', 'a'), f'First merge: {bpe.merges[0]}'\n"
        },
        {
            "name": "编码返回字符串列表",
            "code": "\nbpe = {fn}()\nbpe.train(['low', 'lower', 'lowest'] * 3, num_merges=10)\ntokens = bpe.encode('low')\nassert isinstance(tokens, list), 'encode must return a list'\nassert all(isinstance(t, str) for t in tokens), 'tokens must be strings'\nreconstructed = ''.join(t.replace('</w>', '') for t in tokens)\nassert reconstructed == 'low', f'Reconstruction: {reconstructed}'\n"
        },
        {
            "name": "合并次数越多，词元越少",
            "code": "\nbpe1 = {fn}()\nbpe1.train(['hello'] * 10, num_merges=2)\nbpe2 = {fn}()\nbpe2.train(['hello'] * 10, num_merges=10)\nassert len(bpe2.encode('hello')) <= len(bpe1.encode('hello')), 'More merges should reduce tokens'\n"
        }
    ]
}
