import random
import pandas as pd
from datetime import datetime, timedelta

DEVICE = "DEV-RTR-1001"
INTERFACE = "Gi0"

START = datetime(2025,8,1)
END = datetime(2026,3,31,23,55)

POLL_INTERVAL = timedelta(minutes=5)

device_metrics = []
interface_metrics = []
events = []

device_up = True
active = {}

def event(ts, stat, msg, cleared=False):
    events.append({
        "timestamp": ts,
        "device": DEVICE,
        "interface": INTERFACE,
        "alarm_msg": msg,
        "stat": stat,
        "is_cleared": cleared,
        "threshold_breached": not cleared
    })

def random_device_metrics():
    return {
        "avail":100,
        "c_util":random.uniform(10,70),
        "m_util":random.uniform(10,70),
        "bf_util":random.uniform(5,60)
    }

def random_interface_metrics():
    return {
        "n_avail":100,
        "util":random.uniform(5,70),
        "errs":random.uniform(0,10),
        "discards":random.uniform(0,10),
        "vol":random.uniform(5,70)
    }

t = START

while t <= END:

    if device_up:
        d = random_device_metrics()
        i = random_interface_metrics()
    else:
        d = {"avail":0,"c_util":0,"m_util":0,"bf_util":0}
        i = {"n_avail":0,"util":0,"errs":0,"discards":0,"vol":0}

    # randomly trigger alarms
    if random.random() < 0.015:

        choice = random.choice([
            "device_down","link_down","util","errs",
            "discards","vol","cpu","mem","buf"
        ])

        if choice == "device_down" and device_up:
            device_up=False
            d["avail"]=0
            event(t,"avail","Device Not Reachable")
            active["device"]=True

        elif choice == "link_down":
            i["n_avail"]=0
            event(t,"n_avail","Link Down")
            active["link"]=True

        elif choice == "util":
            i["util"]=random.uniform(85,100)
            event(t,"util","Link Utilization High")
            active["util"]=True

        elif choice == "errs":
            i["errs"]=random.uniform(85,100)
            event(t,"errs","Link Error Rate High")
            active["errs"]=True

        elif choice == "discards":
            i["discards"]=random.uniform(85,100)
            event(t,"discards","Link Discards Rate High")
            active["discards"]=True

        elif choice == "vol":
            i["vol"]=random.uniform(85,100)
            event(t,"vol","Link Throughput High")
            active["vol"]=True

        elif choice == "cpu":
            d["c_util"]=random.uniform(85,100)
            event(t,"c_util","Device CPU Utilization High")
            active["cpu"]=True

        elif choice == "mem":
            d["m_util"]=random.uniform(85,100)
            event(t,"m_util","Device Memory Utilization High")
            active["mem"]=True

        elif choice == "buf":
            d["bf_util"]=random.uniform(85,100)
            event(t,"bf_util","Device Buffer Utilization High")
            active["buf"]=True


    # clear alarms
    for a in list(active.keys()):
        if random.random() < 0.01:

            if a=="device":
                device_up=True
                d["avail"]=100
                event(t,"avail","Device Reachable",True)

            elif a=="link":
                i["n_avail"]=100
                event(t,"n_avail","Link Up",True)

            elif a=="util":
                i["util"]=random.uniform(20,60)
                event(t,"util","Link Utilization Normal",True)

            elif a=="errs":
                i["errs"]=random.uniform(0,10)
                event(t,"errs","Link Errors Rate Normal",True)

            elif a=="discards":
                i["discards"]=random.uniform(0,10)
                event(t,"discards","Link Discards Rate Normal",True)

            elif a=="vol":
                i["vol"]=random.uniform(10,60)
                event(t,"vol","Link Throughput Normal",True)

            elif a=="cpu":
                d["c_util"]=random.uniform(20,60)
                event(t,"c_util","Device CPU Utilization Normal",True)

            elif a=="mem":
                d["m_util"]=random.uniform(20,60)
                event(t,"m_util","Device Memeory Utilization Normal",True)

            elif a=="buf":
                d["bf_util"]=random.uniform(10,40)
                event(t,"bf_util","Device Buffer Utilization Normal",True)

            del active[a]

    device_metrics.append({
        "timestamp":t,
        "device":DEVICE,
        **d
    })

    interface_metrics.append({
        "timestamp":t,
        "device":DEVICE,
        "interface":INTERFACE,
        **i
    })

    t += POLL_INTERVAL


pd.DataFrame(device_metrics).to_csv("metrics_device.csv",index=False)
pd.DataFrame(interface_metrics).to_csv("metrics_interface.csv",index=False)
pd.DataFrame(events).to_csv("events.csv",index=False)

print("6 month dataset generated")