#!/usr/bin/env python3
import re
import sys
from time import perf_counter_ns
from contextlib import contextmanager
from pathlib import Path
from typing import Union, TextIO, Any, NoReturn

import shutil

_TFile = Union[TextIO, Path, str]


class Timing:
    def __init__(self, ns, *, context=None, label=None):
        self.ns = int(ns)
        self.micro = round(self.ns / 1_000, 1)
        self.ms = round(self.ns / 1_000_000, 2)
        self.s = self.ns / 1_000_000_000
        self.per_sec = 1_000_000_000 // self.ns
        self.context = context
        self.label = label
    
    def __repr__(self):
        return f'Seconds: {self.s: <15,.10f} | Mili: {self.ms: >7,} | Micro: {self.micro: >12,} | Nano: {self.ns: >12,} | Per-second: {self.per_sec: >7,.0f}'
    
    @classmethod
    def from_string(cls, line: str) -> 'Timing':
        # self.validate_with_openapi('device_event_blocked_traffic') | Seconds: 0.0180272800 | Mili: 18.03 | Micro: 18,027.3 | Nano: 18,027,280 | 55 per second
        ns_match = re.search(r'Nano: ([\d,]+)', line)
        if not ns_match:
            raise ValueError(f'Could not parse line, missing "Nano: ..." | {line=!r}')
        context_match = re.search(r'(.+)(?= \| Seconds)', line)
        timing = cls(ns_match.group(1).replace(',', ''),
                      context=context_match.group(1) if context_match else None)
        return timing

    def to_row(self,
               col_width: int = None,
               *,
               avg: 'Timing' = None,
               context=False,
               format = '{sec} | {mili} | {micro} | {ns} | {per_sec} | {percent_of_avg} | {context}') -> str:
        if context or col_width is None:
            col_width = 0
        _separators = re.sub(r'\s*{[\w]+}\s*', '', format)
        _separator = _separators[0] # '|'
        if _separator * len(_separators) != _separators:
            raise ValueError(f'Inconsistent separators in format string: {format=!r}')
        
        sec = f'{self.s:,.10f}'.rjust(col_width - 3)
        mili = f'{self.ms:,}'.rjust(col_width - 3)
        micro = f'{self.micro:,}'.rjust(col_width - 3)
        ns = f'{self.ns:,}'.rjust(col_width - 3)
        per_sec = f'{self.per_sec:,}'.rjust(col_width - 3)
        info = {'sec': sec, 'mili': mili, 'micro': micro, 'ns':ns, 'per_sec':per_sec}
        
        if avg:
            percent_of_avg = f'{(self.ns / avg.ns):,.0%}'.rjust(col_width - 3)
            info['percent_of_avg'] = percent_of_avg
        else:
            format = format.replace('{percent_of_avg}', '')
        
        if context and self.context:
            info['context'] = self.context
        else:
            format = format.replace('{context}', '')
        breakpoint()
        format = re.sub(f'\s*{_separator}\s*(?={_separator})', '', format)
        row = format.format(**info)
        return row
    
    def out(self, file: _TFile, *, context=False) -> None:
        row = self.to_row(col_width=0,
                          context=context,
                          format='{context} | Seconds: {sec} | Mili: {mili} | Micro: {micro} | Nano: {ns} | {per_sec} per second')
        try:
            with open(file, 'a') as stream:
                stream.write(row + '\n')
        except (FileNotFoundError, TypeError, PermissionError):
            print(row, file=file)
        

class Profile:
    def __init__(self,
                 timings: list[Timing],
                 *,
                 file_path = None
                 ):
        ns_sum = 0
        timings_count = 0
        self.timings = []
        for timing in sorted(timings, key=lambda prof: prof.ns):
            ns_sum += timing.ns
            timings_count += 1
            self.timings.append(timing)
        
        self.ns_sum = ns_sum
        self.avg = Timing(round(self.ns_sum / timings_count, 2))
        deviation_sum = 0
        for nano in [timing.ns for timing in self.timings]:
            deviation = (nano - self.avg.ns) ** 2
            deviation_sum += deviation
        self.variance: int = int(deviation_sum / (timings_count - 1))
        self.stdev = Timing(int(self.variance ** 0.5))
        self.median = self.timings[timings_count // 2]
        
        self._file_path = file_path
    
    @classmethod
    def from_file(cls, file_path) -> 'Profile':
        timings = []
        with open(file_path) as file:
            for line in file.readlines():
                timing = Timing.from_string(line.strip())
                timings.append(timing)
        return cls(timings)
    
    def trim_ends(self, index: int) -> 'Profile':
        """Return s a new Profile with the extreme values beyond `index` removed."""
        if index <= 0:
            return self
        return Profile(self.timings[index:-index])

    def print_table(self,
                    *,
                    context=False) -> NoReturn:
        """Prints a table representation of the Profile."""
        if self._file_path:
            print(f'\x1b[97;1m{self._file_path}\x1b[0m')
        
        # Build rows data
        row_format = '{context}\n{sec} | {mili} | {micro} | {ns} | {per_sec} | {percent_of_avg}'
        timings_rows = {}
        for timing_name, timing_index in [('Fastest', 0),
                                          ('2nd Fastest', 1),
                                          ('Slowest', -1),
                                          ('2nd Slowest', -2),
                                          ]:
            timing = self.timings[timing_index]
            row = timing.to_row(avg=self.avg, context=context, format=row_format)
            timings_rows[timing_name] = row
            # formatted_timings_rows.append(f'| {timing_name.ljust(col_width - 3)}{row} |')
    
        # timings_rows.append('\x1b[2m' + '-' * (col_width // 2) + f' Statistics ' + '-' * (col_width * len(columns) - 12 - col_width // 2) + '\x1b[0m')
        stats_rows = {}
        for stat_name, stat_timing in [('Average', self.avg),
                                        ('Average[1:-1]', self.trim_ends(1).avg),
                                        ('Average[2:-2]', self.trim_ends(2).avg),
                                        ('Median', self.median),
                                        ('Std Dev', self.stdev),
                                        ]:
            row = stat_timing.to_row(avg=self.avg, context=context, format=row_format)
            stats_rows[stat_name] = row
            # formatted_timings_rows.append(f'| {stat_name.ljust(col_width - 3)}{row} |')
        
        # Build table
        columns = ["Run", "Seconds", "Mili", "Micro", "Nano", "Per-second", "% of Avg"]
        term_width = shutil.get_terminal_size()[0]
        # col_width = term_width // len(columns) - 8
        col_width = max(max(len(line) for line in row.splitlines()) for row in timings_rows + stats_rows)
        print(f'{col_width = } | {term_width = } | {col_width * len(columns) = }')
        # for i, column in enumerate(columns):
        #     whitespace = ' ' * ((col_width - len(column)) // 2)
        #     columns[i] = whitespace + column + whitespace
        formatted_columns = [column.ljust(col_width - 3) for column in columns]
        horizonal_div = '-' * (col_width * len(columns))
        print('\n' + horizonal_div)
        print('| ' + ' | '.join(map(lambda col: f'\x1b[1m{col}\x1b[0m', formatted_columns)) + ' |', end='')
        print('\n' + horizonal_div)

        formatted_rows = []
        for row_name, row in timings_rows.items():
            formatted_rows.append(f'| {row_name.ljust(col_width - 3)}{row} |')
        formatted_rows.append('\x1b[2m' + '-' * (col_width // 2) + f' Statistics ' + '-' * (col_width * len(columns) - 12 - col_width // 2) + '\x1b[0m')
        for row_name, row in stats_rows.items():
            formatted_rows.append(f'| {stat_name.ljust(col_width - 3)}{row} |')
        formatted_rows.extend([horizonal_div,
                     f'Variance        {self.variance:,} ns\n'])
        
        table = '\n'.join(timings_rows)
        print(table)


@contextmanager
def timeit(context='',
           *,
           quiet=False,
           timing_includes_exception_handling=True,
           label=None,
           file: _TFile = sys.stdout) -> Timing:
    timing = None
    start = perf_counter_ns()
    exception_occurred = False
    try:
        yield
    except:
        exception_occurred = True
        raise
    finally:
        if not exception_occurred or timing_includes_exception_handling:
            timing = Timing(perf_counter_ns() - start, context=context, label=label)
            if not quiet:
                timing.out(file=file, context=True)  # ok even if context is empty
        if not exception_occurred:
            return timing


def stats(file_path, *, context=False):
    profile = Profile.from_file(file_path)
    profile.print_table(context=context)


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    argparser = ArgumentParser()
    argparser.add_argument('profiling_file')
    argparser.add_argument('-c', '--context', default=False, action='store_true')
    parsed_args = argparser.parse_args()
    stats(parsed_args.profiling_file, context=parsed_args.context)
