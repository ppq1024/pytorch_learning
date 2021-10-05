'''
    Copyright (c) 2021 PPQ

    This file is part of pytorch learning (pl).

    pl is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    pl is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with pl.  If not, see <https://www.gnu.org/licenses/>.
'''

from typing import List, Dict


def parseArg(args:List[str]) -> Dict[str, str]:
    result:dict[str, str] = {}
    result.setdefault('new_model', 'false')
    for arg in args:
        entry = arg.split('=', 1)
        if (entry[0] == '-n'):
            result['new_model'] = 'true'
        elif (entry[0] == '-m'):
            m = entry[1].split('.', 1)
            result['model_type'] = m[0]
            result['model_name'] = m[1]
        else:
            result[entry[0]] = entry[1]
    return result
