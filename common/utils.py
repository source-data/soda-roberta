import re
from datetime import datetime


def innertext(xml):
    return "".join([t for t in xml.itertext()])


def cleanup(text):
    text = re.sub('[\r\n\t]', ' ', text)
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[–—‐−]', '-', text) # controversial!!!
    text = re.sub(r'^[Aa]bstract', '', text)
    return text


def now():
    now_str = datetime.now().isoformat()
    now_str.replace(':', '-')
    return now_str

# -*- coding: utf-8 -*-
# From https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
# The MIT License (MIT)
# Copyright (c) 2016 Vladimir Ignatev
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software
# is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * (count+1) / float(total)))

    percents = round(100.0 * (count + 1) / float(total))
    bar = '█' * filled_len + '-' * (bar_len - filled_len)

    print('\r[%s] %s%s ...%s' % (bar, percents, '%', status), end="\r", flush=True)
