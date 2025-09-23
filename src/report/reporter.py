# ** coding: utf-8 **
"""
@Author: Zhihong Li, Rongrong Lin
@date: 2024/11/25
@description: Reporter.
@version: 2.0
"""

def mkdir(path: str):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


class Reporter:
    def __init__(self, opts, filename: str = None) -> None:
        self.static_opts = opts
        self.dynamic = []
        save_dir=self.static_opts.save_dir
        self.filename = save_dir + '/report.txt' if filename is None else filename
        mkdir(save_dir)

    def save_opts(self):
        with open(self.filename, 'w+') as f:
            f.write('------------------------')
            for k, v in self.static_opts.__dict__.items():
                f.write(k + ': ' + str(v) + '\n')
            f.write('------------------------')

    def save_inf(self, inf: str):
        with open(self.filename, 'a') as f:
            f.write('\n\n')
            f.write(inf)
