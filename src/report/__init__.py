from report.reporter import Reporter


def init_settings(opts, filename):
    rep= Reporter(opts, filename)
    rep.save_opts()
    return rep
