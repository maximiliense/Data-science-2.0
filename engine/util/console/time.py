import time
import datetime
import pytz

start_time = None


def get_start_datetime():
    global start_time
    start_time = time.time()
    d = datetime.datetime.utcfromtimestamp(start_time)
    tz = pytz.timezone('UTC')
    d = tz.localize(d)
    d = d.astimezone(pytz.timezone('Europe/Paris'))
    return d
