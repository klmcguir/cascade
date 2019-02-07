import flow
import pool
import numpy as np

days = flow.metadata.DateSorter.frommeta(mice=['OA27'], tags=None)
css = ['plus', 'minus', 'neutral']

for day in days:
    if not np.any([True for s in day.tags if s == 'naive']):
        for cs in css:
            pool.calc.driven.visually(day, cs)
