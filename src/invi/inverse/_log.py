"""Functions used to log the results of the optimisations."""

import time as _time
import datetime as _dt

import fnc as _fnc

#-----------------------------------------------------------------------------

def log_print(x, file, flush=True):
    print(x, file=file, flush=flush)


def log_print_dict(x, file):
    if not isinstance(x, dict):
        raise ValueError(f"{x} is not a dict.")

    length = []
    for key in x.keys():
        length += [len(key)]
    space = max(length)

    for key in x.keys():
        log_print(f"{key:>{space}}: {x[key]}", file)

#-----------------------------------------------------------------------------

class NumberEvaluations:
    def __init__(self):
        self.n = -1

#-----------------------------------------------------------------------------

def log_info_evaluation(x, mu_eval, mu_x1, n_eval, options, log_file):
    #Space required by the number of maxfev
    space = len(str(options['maxfev']))

    header = f"{'N':{space+1}}| q     | Md    | ad    | ah    | --> [Min]"
    line = f"{n_eval:{space}} | {x[0]:0.3f} | {x[1]:0.3f} | {x[2]:0.3f} | {x[3]:0.3f} | --> {mu_eval/mu_x1:0.7f}"

    #Print header every 20 lines
    if n_eval % 20 == 0:
        log_print(header, log_file)
    log_print(line, log_file)


def log_header(options, file):
    start_date = _dt.datetime.now().strftime('%Y/%b/%d %H:%M:%S')
    start_time = _time.perf_counter()

    _fnc.info.system(file=file)

    log_print('\nOptions', file)
    log_print('-------', file)
    log_print_dict(options, file)

    log_print('\nEvaluations', file)
    log_print('-----------', file)

    return start_date, start_time


def log_footer(res, dim, sample_icrs, mu_eval_x1, start_date, start_time, file):
    log_print('\nResults Nelder-Mead', file)
    log_print('-------------------', file)
    log_print(res, file)

    log_print('\nData', file)
    log_print('----', file)
    log_print(f'  Number dim: {dim:_}', file)
    log_print(f'Number stars: {len(sample_icrs[0]):_}', file)

    str_array = _fnc.numeric.print_array(res.x, precision=15, verbose=False)
    log_print(f'  mu_eval_x1: {mu_eval_x1*1_000.0:0.5f} [mrad]', file)
    log_print(f'   mu_eval_x: {res.fun*1_000.0:0.5f} [mrad]', file)
    log_print(f'           x: {str_array}', file)

    log_print('\nTime', file)
    log_print('----', file)
    log_print(f'  Start: {start_date}', file)
    end_date = _dt.datetime.now().strftime('%Y/%b/%d %H:%M:%S')
    end_time = _time.perf_counter()
    hr_time = _fnc.utils.human_readable.time(end_time - start_time)
    log_print(f'    End: {end_date}', file)
    log_print(f'Elapsed: {hr_time}', file)

#-----------------------------------------------------------------------------

def log_info_optimisation(success, x, mu_eval, mu_x1, N_opt, options, log_file):
    #Space required by the number of N_opts
    space = len(str(options['N_opts']))

    #Convergence result
    if success:
        result = 'Success'
    else:
        result = 'Failure'

    header = f"{'N':{space+1}}| Convergence | q     | Md    | ad    | ah    | --> [Min]"
    line = f"{N_opt:{space}} | {result}     | {x[0]:0.3f} | {x[1]:0.3f} | {x[2]:0.3f} | {x[3]:0.3f} | --> {mu_eval/mu_x1:0.7f}"

    #Print header every 20 lines
    if N_opt % 20 == 0:
        log_print(header, log_file)

    log_print(line, log_file)


def log_header_multiple(options, file):
    start_date = _dt.datetime.now().strftime('%Y/%b/%d %H:%M:%S')
    start_time = _time.perf_counter()

    _fnc.info.system(file=file)

    log_print('\nOptions', file)
    log_print('-------', file)
    log_print_dict(options, file)

    log_print('\nResults', file)
    log_print('-------', file)

    return start_date, start_time


def log_footer_multiple(dim, samples_icrs, start_date, start_time, file):
    log_print('\nData', file)
    log_print('----', file)
    log_print(f'  Number dim: {dim:_}', file)
    log_print(f'Number stars: {len(samples_icrs[0][0]):_}', file)

    log_print('\nTime', file)
    log_print('----', file)
    log_print(f'  Start: {start_date}', file)
    end_date = _dt.datetime.now().strftime('%Y/%b/%d %H:%M:%S')
    end_time = _time.perf_counter()
    hr_time = _fnc.utils.human_readable.time(end_time - start_time)
    log_print(f'    End: {end_date}', file)
    log_print(f'Elapsed: {hr_time}', file)

#-----------------------------------------------------------------------------
