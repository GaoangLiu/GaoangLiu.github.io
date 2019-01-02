from urllib.request import urlopen
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import re
import requests
import random
import datetime

# link = "http://www.pythonscraping.com/pages/page3.html"
link = "https://www.douban.com/photos/photo/2245077096/"
link = "https://www.douban.com/photos/photo/2216271581/"
html = requests.get(link, 'html.parser')
print(html)
bsojb = BeautifulSoup(html.text, features="lxml")

# for name in bsojb.findAll("img", {"src":re.compile(".*\.jpg")}):
for name in bsojb.findAll('link'):
    print(name)
    # print(name.findAll('img'))


pages = set()
cter = 0


def get_links(url):
    global pages
    global cter
    html = urlopen("http://en.wikipedia.org" + url)
    bsojb = BeautifulSoup(html, features="lxml")
    try:
        print(bsojb.h1.get_text())
        print(bsojb.find(id="mw-content-text").findAll("p")[0])
        print(bsojb.find(id="ca-edit").find("span").find("a").attrs['href'])
    except AttributeError:
        print("Dont worry, just minor mistakes")

    for link in bsojb.findAll("a", href=re.compile("^(/wiki/)")):
        if 'href' in link.attrs:
            if link.attrs['href'] not in pages:
                new_page = link.attrs['href']
                print(new_page)
                pages.add(new_page)
                get_links(new_page)


random.seed(datetime.datetime.now())


def get_internal_links(bsojb, includeurl):
    includeurl = urlparse(includeurl).scheme + "://" + \
        urlparse(includeurl).netloc
    internal_links = []

    for link in bsojb.findAll(
            "a", href=re.compile(
            "^(/|.*" + includeurl + ")")):
        if link.attrs['href'] is not None:
            if link.attrs['href'] not in internal_links:
                if link.attrs['href'].startswith("/"):
                    internal_links.append(includeurl + link.attrs['href'])
                else:
                    internal_links.append(link.attrs['href'])

    return internal_links


def get_external_links(bsojb, excludeurl):
    external_links = []
    for link in bsojb.findAll("a", href=re.compile(
            "^(http|www)((?!" + excludeurl + ").)*$")):
        if link.attrs['href'] is not None:
            if link.attrs['href'] not in external_links:
                external_links.append(link.attrs['href'])
    return external_links


def split_address(addr):
    return addr.replace("http://", "").split("/")


def get_random_external_link(starting_page)	:
    html = urlopen(starting_page)
    bsojb = BeautifulSoup(html, features="lxml")
    external_links = get_external_links(bsojb, urlparse(starting_page).netloc)
    if len(external_links) == 0:
        print("No external links found.")
        domain = urlparse(starting_page).scheme + "://" + \
            urlparse(starting_page).netloc
        internal_links = get_internal_links(bsojb, domain)
        return get_random_external_link(
            internal_links[random.randint(0, len(internal_links) - 1)])
    else:
        return external_links[random.randint(0, len(external_links) - 1)]


def follow_external_only(starting_page):
    ext_url = get_random_external_link(starting_page)
    print("Random external link is: " + ext_url)
    follow_external_only(ext_url)


follow_external_only("http://oreilly.com")
