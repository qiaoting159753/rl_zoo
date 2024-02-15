import logging
from tqdm import tqdm, trange
import time

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setStream(tqdm) # <-- important
log.addHandler(handler)

for i in trange(100):
    if i % 10 == 0:
        a = 0.5
        b = 2
        c = 3
        log.info(f"Evaluation: {a/b} at {c} \n")
        # log.info("\nHalf-way there!\n")
    time.sleep(0.1)