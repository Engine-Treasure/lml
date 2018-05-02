import logging
import os
import re
import sys
from urllib import urlretrieve
from urlparse import urljoin

import fire
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",  # display date
    datefmt="[%Y-%m-%d %H:%M:%S]"
)

LINK_PATTERN = re.compile('href="(.*?)"')


class BatchDownloader(object):
    def __init__(self):
        pass

    def uci(self, url):
        dataset_name = url.strip("/").split("/")[-1]

        if not os.path.exists(dataset_name):
            os.makedirs(dataset_name)

        html = requests.get(url).text
        inner_link = filter(
            lambda t: not (t.startswith("http") or t.startswith("?") or t.startswith("#")),
            LINK_PATTERN.findall(html)
        )

        logging.info("Start")
        for ilink in inner_link[2:]:
            try:
                # r = requests.get(url + ilink)
                # with open(os.path.join(dataset_name, ilink), "wb") as f:
                #     f.write(r.text)
                urlretrieve(urljoin(url, ilink), os.path.join(dataset_name, ilink))
                logging.info("{file} downloaded.".format(file=ilink))
            except Exception as e:
                logging.error(e)
                continue
        logging.info("Complete")


if __name__ == '__main__':
    fire.Fire(BatchDownloader)