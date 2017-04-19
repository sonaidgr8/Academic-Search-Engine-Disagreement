#! /usr/bin/env python
"""
This module provides classes for querying Google Scholar and parsing
returned results. It currently *only* processes the first results
page. It is not a recursive crawler.
"""


import optparse
import os
import sys
import re
import logging
import sqlalchemy
from rdflib.graph import Graph
import skos
import numpy as np
import itertools
from fuzzywuzzy import fuzz
from sklearn.metrics import jaccard_similarity_score
import requests
import json
from my_keys import MY_SOCPUS_API_KEY, My_MAS_API_KEY1, My_MAS_API_KEY2
from selenium import webdriver
import selenium.webdriver.support.ui as ui
from selenium.webdriver.common.keys import Keys
#from selenium.webdriver.remote.webelement.WebElement import Keys
import time
try:
    # Try importing for Python 3
    # pylint: disable-msg=F0401
    # pylint: disable-msg=E0611
    from urllib.request import HTTPCookieProcessor, Request, build_opener
    from urllib.parse import quote, unquote
    from http.cookiejar import MozillaCookieJar
except ImportError:
    # Fallback for Python 2
    from urllib2 import Request, build_opener, HTTPCookieProcessor
    from urllib import quote, unquote
    from cookielib import MozillaCookieJar

# Import BeautifulSoup -- try 4 first, fall back to older
try:
    from bs4 import BeautifulSoup
except ImportError:
    try:
        from BeautifulSoup import BeautifulSoup
    except ImportError:
        print('We need BeautifulSoup, sorry...')
        sys.exit(1)

# Support unicode in both Python 2 and 3. In Python 3, unicode is str.
if sys.version_info[0] == 3:
    unicode = str # pylint: disable-msg=W0622
    encode = lambda s: unicode(s) # pylint: disable-msg=C0103
else:
    def encode(s):
        if isinstance(s, basestring):
            return s.encode('utf-8') # pylint: disable-msg=C0103
        else:
            return str(s)


class Error(Exception):
    """Base class for any Scholar error."""


class FormatError(Error):
    """A query argument or setting was formatted incorrectly."""


class QueryArgumentError(Error):
    """A query did not have a suitable set of arguments."""


class ScholarConf(object):
    """Helper class for global settings."""

    VERSION = '2.10'
    LOG_LEVEL = 1
    MAX_PAGE_RESULTS = 8 # Current maximum for per-page results
    GS_SCHOLAR_SITE = 'http://scholar.google.com'
    SS_SCHOLAR_SITE = 'http://semanticscholar.org'
    
    #USER_AGENT = 'Mozilla/5.0 (X11; U; FreeBSD i386; en-US; rv:1.9.2.9) Gecko/20100913 Firefox/3.6.9'
    # Let's update at this point (3/14):
    USER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64; rv:27.0) Gecko/20100101 Firefox/27.0'
    #USER_AGENT = 'Mozilla/5.0 (X11; U; Linux x86_64; en-us) AppleWebKit/531.2+ (KHTML, like Gecko) Version/5.0 Safari/531.2+'

    # If set, we will use this file to read/save cookies to enable
    # cookie use across sessions.
    COOKIE_JAR_FILE = None

class ScholarUtils(object):
    """A wrapper for various utensils that come in handy."""

    LOG_LEVELS = {'error': 1,
                  'warn':  2,
                  'info':  3,
                  'debug': 4}

    @staticmethod
    def ensure_int(arg, msg=None):
        try:
            return int(arg)
        except ValueError:
            raise FormatError(msg)

    @staticmethod
    def log(level, msg):
        if level not in ScholarUtils.LOG_LEVELS.keys():
            return
        if ScholarUtils.LOG_LEVELS[level] > ScholarConf.LOG_LEVEL:
            return
        sys.stderr.write('[%5s]  %s' % (level.upper(), msg + '\n'))
        sys.stderr.flush()

class ScholarArticle(object):
    """
    A class representing articles listed on Google Scholar.  The class
    provides basic dictionary-like behavior.
    """
    def __init__(self):
        # The triplets for each keyword correspond to (1) the actual
        # value, (2) a user-suitable label for the item, and (3) an
        # ordering index:
        self.attrs = {
            'title':         [None, 'Title',          0],
            'url':           [None, 'URL',            1],
            'year':          [None, 'Year',           2],
            'num_citations': [0,    'Citations',      3],
            'num_versions':  [0,    'Versions',       4],
            'cluster_id':    [None, 'Cluster ID',     5],
            'url_pdf':       [None, 'PDF link',       6],
            'url_citations': [None, 'Citations list', 7],
            'url_versions':  [None, 'Versions list',  8],
            'url_citation':  [None, 'Citation link',  9],
            'excerpt':       [None, 'Excerpt',       10],
        }

        # The citation data in one of the standard export formats,
        # e.g. BibTeX.
        self.citation_data = None

    def __getitem__(self, key):
        if key in self.attrs:
            return self.attrs[key][0]
        return None

    def __len__(self):
        return len(self.attrs)

    def __setitem__(self, key, item):
        if key in self.attrs:
            self.attrs[key][0] = item
        else:
            self.attrs[key] = [item, key, len(self.attrs)]

    def __delitem__(self, key):
        if key in self.attrs:
            del self.attrs[key]

    def set_citation_data(self, citation_data):
        self.citation_data = citation_data

    def as_txt(self):
        # Get items sorted in specified order:
        items = sorted(list(self.attrs.values()), key=lambda item: item[2])
        # Find largest label length:
        max_label_len = max([len(str(item[1])) for item in items])
        fmt = '%%%ds %%s' % max_label_len
        res = []
        for item in items:
            if item[0] is not None:
                res.append(fmt % (item[1], item[0]))
        return '\n'.join(res)

    def as_csv(self, header=False, sep='|'):
        # Get keys sorted in specified order:
        keys = [pair[0] for pair in \
                sorted([(key, val[2]) for key, val in list(self.attrs.items())],
                       key=lambda pair: pair[1])]
        res = []
        if header:
            res.append(sep.join(keys))
        res.append(sep.join([unicode(self.attrs[key][0]) for key in keys]))
        return '\n'.join(res)

    def as_citation(self):
        """
        Reports the article in a standard citation format. This works only
        if you have configured the querier to retrieve a particular
        citation export format. (See ScholarSettings.)
        """
        return self.citation_data or ''


class ScholarArticleParser(object):
    """
    ScholarArticleParser can parse HTML document strings obtained from
    Google Scholar. This is a base class; concrete implementations
    adapting to tweaks made by Google over time follow below.
    """
    def __init__(self, choice, site=None):
        self.soup = None
        self.article = None
        if choice == 'gs':
            self.site = ScholarConf.GS_SCHOLAR_SITE
        elif choice == 'ss':
            self.site = ScholarConf.SS_SCHOLAR_SITE
        self.year_re = re.compile(r'\b(?:20|19)\d{2}\b')

    def handle_article(self, art):
        """
        The parser invokes this callback on each article parsed
        successfully.  In this base class, the callback does nothing.
        """

    def handle_num_results(self, num_results):
        """
        The parser invokes this callback if it determines the overall
        number of results, as reported on the parsed results page. The
        base class implementation does nothing.
        """

    def parse(self, html, choice):
        """
        This method initiates parsing of HTML content, cleans resulting
        content as needed, and notifies the parser instance of
        resulting instances via the handle_article callback.
        """
        self.soup = BeautifulSoup(html)

        # This parses any global, non-itemized attributes from the page.
        if choice == 'gs' :
            self._parse_globals()
            # Now parse out listed articles:
            for art in self.soup.findAll(ScholarArticleParser._gs_tag_results_checker):
                self._gs_parse_article(art)
                self._clean_article()
                if self.article['title']:
                    self.handle_article(self.article)
        
        if choice == 'ss' :
            for art in self.soup.findAll(ScholarArticleParser._ss_tag_results_checker):
                self._ss_parse_article(art)
                self._clean_article()
                if self.article['title']:
                    self.handle_article(self.article)
                    

    def _clean_article(self):
        """
        This gets invoked after we have parsed an article, to do any
        needed cleanup/polishing before we hand off the resulting
        article.
        """
        if self.article['title']:
            self.article['title'] = self.article['title'].strip()

    def _parse_globals(self):
        tag = self.soup.find(name='div', attrs={'id': 'gs_ab_md'})
        if tag is not None:
            raw_text = tag.findAll(text=True)
            # raw text is a list because the body contains <b> etc
            if raw_text is not None and len(raw_text) > 0:
                try:
                    num_results = raw_text[0].split()[1]
                    # num_results may now contain commas to separate
                    # thousands, strip:
                    num_results = num_results.replace(',', '')
                    num_results = int(num_results)
                    self.handle_num_results(num_results)
                except (IndexError, ValueError):
                    pass

    def _parse_article(self, div):
        self.article = ScholarArticle()

        for tag in div:
            if not hasattr(tag, 'name'):
                continue

            if tag.name == 'div' and self._tag_has_class(tag, 'gs_rt') and \
                    tag.h3 and tag.h3.a:
                self.article['title'] = ''.join(tag.h3.a.findAll(text=True))
                self.article['url'] = self._path2url(tag.h3.a['href'])
                if self.article['url'].endswith('.pdf'):
                    self.article['url_pdf'] = self.article['url']

            if tag.name == 'font':
                for tag2 in tag:
                    if not hasattr(tag2, 'name'):
                        continue
                    if tag2.name == 'span' and \
                       self._tag_has_class(tag2, 'gs_fl'):
                        self._parse_links(tag2)

    def _parse_links(self, span):
        for tag in span:
            if not hasattr(tag, 'name'):
                continue
            if tag.name != 'a' or tag.get('href') is None:
                continue

            if tag.get('href').startswith('/scholar?cites'):
                if hasattr(tag, 'string') and tag.string.startswith('Cited by'):
                    self.article['num_citations'] = \
                        self._as_int(tag.string.split()[-1])

                # Weird Google Scholar behavior here: if the original
                # search query came with a number-of-results limit,
                # then this limit gets propagated to the URLs embedded
                # in the results page as well. Same applies to
                # versions URL in next if-block.
                self.article['url_citations'] = \
                    self._strip_url_arg('num', self._path2url(tag.get('href')))

                # We can also extract the cluster ID from the versions
                # URL. Note that we know that the string contains "?",
                # from the above if-statement.
                args = self.article['url_citations'].split('?', 1)[1]
                for arg in args.split('&'):
                    if arg.startswith('cites='):
                        self.article['cluster_id'] = arg[6:]

            if tag.get('href').startswith('/scholar?cluster'):
                if hasattr(tag, 'string') and tag.string.startswith('All '):
                    self.article['num_versions'] = \
                        self._as_int(tag.string.split()[1])
                self.article['url_versions'] = \
                    self._strip_url_arg('num', self._path2url(tag.get('href')))

            if tag.getText().startswith('Import'):
                self.article['url_citation'] = self._path2url(tag.get('href'))


    @staticmethod
    def _tag_has_class(tag, klass):
        """
        This predicate function checks whether a BeatifulSoup Tag instance
        has a class attribute.
        """
        res = tag.get('class') or []
        if type(res) != list:
            # BeautifulSoup 3 can return e.g. 'gs_md_wp gs_ttss',
            # so split -- conveniently produces a list in any case
            res = res.split()
        return klass in res

    @staticmethod
    def _ss_tag_results_checker(tag):
        return tag.name == 'article' \
            and ScholarArticleParser._tag_has_class(tag, 'search-result')
            
    @staticmethod
    def _gs_tag_results_checker(tag):
        return tag.name == 'div' \
            and ScholarArticleParser._tag_has_class(tag, 'gs_r')
            
    @staticmethod
    def _as_int(obj):
        try:
            return int(obj)
        except ValueError:
            return None

    def _path2url(self, path):
        """Helper, returns full URL in case path isn't one."""
        if path.startswith('http://'):
            return path
        if not path.startswith('/'):
            path = '/' + path
        return self.site + path

    def _strip_url_arg(self, arg, url):
        """Helper, removes a URL-encoded argument, if present."""
        parts = url.split('?', 1)
        if len(parts) != 2:
            return url
        res = []
        for part in parts[1].split('&'):
            if not part.startswith(arg + '='):
                res.append(part)
        return parts[0] + '?' + '&'.join(res)


class ScholarArticleParser120726(ScholarArticleParser):
    """
    This class reflects update to the Scholar results page layout that
    Google made 07/26/12.
    """
    def _ss_parse_article(self, art):
        self.article = ScholarArticle()
        for tag in art:
            if tag.name == 'header' and self._tag_has_class(tag, 'search-result-header') :
                #if not hasattr(subtag, 'name'):
                    #continue
                """if str(tag).lower().find('.pdf'):
                    if tag.find('div', {'class': 'gs_ttss'}):
                        self._parse_links(tag.find('div', {'class': 'gs_ttss'}))"""
                atag = tag.div.div.a
                self.article['title'] = ''.join(atag.findAll(text=True))
                self.article['url'] = self._path2url(atag['href'])
                #ytag = tag.find('ul', {'class': 'subhead'})
                #li_list = ytag.findAll('li',text=True)
            
                  
    def _gs_parse_article(self, div):
        self.article = ScholarArticle()

        for tag in div:
            if not hasattr(tag, 'name'):
                continue
            if str(tag).lower().find('.pdf'):
                if tag.find('div', {'class': 'gs_ttss'}):
                    self._parse_links(tag.find('div', {'class': 'gs_ttss'}))

            if tag.name == 'div' and self._tag_has_class(tag, 'gs_ri'):
                # There are (at least) two formats here. In the first
                # one, we have a link, e.g.:
                #
                # <h3 class="gs_rt">
                #   <a href="http://dl.acm.org/citation.cfm?id=972384" class="yC0">
                #     <b>Honeycomb</b>: creating intrusion detection signatures using
                #        honeypots
                #   </a>
                # </h3>
                #
                # In the other, there's no actual link -- it's what
                # Scholar renders as "CITATION" in the HTML:
                #
                # <h3 class="gs_rt">
                #   <span class="gs_ctu">
                #     <span class="gs_ct1">[CITATION]</span>
                #     <span class="gs_ct2">[C]</span>
                #   </span>
                #   <b>Honeycomb</b> automated ids signature creation using honeypots
                # </h3>
                #
                # We now distinguish the two.
                try:
                    atag = tag.h3.a
                    self.article['title'] = ''.join(atag.findAll(text=True))
                    self.article['url'] = self._path2url(atag['href'])
                    if self.article['url'].endswith('.pdf'):
                        self.article['url_pdf'] = self.article['url']
                except:
                    # Remove a few spans that have unneeded content (e.g. [CITATION])
                    for span in tag.h3.findAll(name='span'):
                        span.clear()
                    self.article['title'] = ''.join(tag.h3.findAll(text=True))

                if tag.find('div', {'class': 'gs_a'}):
                    year = self.year_re.findall(tag.find('div', {'class': 'gs_a'}).text)
                    self.article['year'] = year[0] if len(year) > 0 else None

                if tag.find('div', {'class': 'gs_fl'}):
                    self._parse_links(tag.find('div', {'class': 'gs_fl'}))

                if tag.find('div', {'class': 'gs_rs'}):
                    # These are the content excerpts rendered into the results.
                    raw_text = tag.find('div', {'class': 'gs_rs'}).findAll(text=True)
                    if len(raw_text) > 0:
                        raw_text = ''.join(raw_text)
                        raw_text = raw_text.replace('\n', '')
                        self.article['excerpt'] = raw_text


class ScholarQuery(object):
    """
    The base class for any kind of results query we send to Scholar.
    """
    def __init__(self):
        self.url = None

        # The number of results requested from Scholar -- not the
        # total number of results it reports (the latter gets stored
        # in attrs, see below).
        self.num_results = ScholarConf.MAX_PAGE_RESULTS

        # Queries may have global result attributes, similar to
        # per-article attributes in ScholarArticle. The exact set of
        # attributes may differ by query type, but they all share the
        # basic data structure:
        self.attrs = {}

    def set_num_page_results(self, num_page_results):
        msg = 'maximum number of results on page must be numeric'
        self.num_results = ScholarUtils.ensure_int(num_page_results, msg)

    def get_url(self):
        """
        Returns a complete, submittable URL string for this particular
        query instance. The URL and its arguments will vary depending
        on the query.
        """
        return None

    def _add_attribute_type(self, key, label, default_value=None):
        """
        Adds a new type of attribute to the list of attributes
        understood by this query. Meant to be used by the constructors
        in derived classes.
        """
        if len(self.attrs) == 0:
            self.attrs[key] = [default_value, label, 0]
            return
        idx = max([item[2] for item in self.attrs.values()]) + 1
        self.attrs[key] = [default_value, label, idx]

    def __getitem__(self, key):
        """Getter for attribute value. Returns None if no such key."""
        if key in self.attrs:
            return self.attrs[key][0]
        return None

    def __setitem__(self, key, item):
        """Setter for attribute value. Does nothing if no such key."""
        if key in self.attrs:
            self.attrs[key][0] = item

    def _parenthesize_phrases(self, query):
        """
        Turns a query string containing comma-separated phrases into a
        space-separated list of tokens, quoted if containing
        whitespace. For example, input

          'some words, foo, bar'

        becomes

          '"some words" foo bar'

        This comes in handy during the composition of certain queries.
        """
        if query.find(',') < 0:
            return query
        phrases = []
        for phrase in query.split(','):
            phrase = phrase.strip()
            if phrase.find(' ') > 0:
                phrase = '"' + phrase + '"'
            phrases.append(phrase)
        return ' '.join(phrases)


class SearchScholarQuery(ScholarQuery):
    """
    This version represents the search query parameters the user can
    configure on the Scholar website, in the advanced search options.
    """
    SCHOLAR_QUERY_URL = ''
    urlargs = {}
    GS_SCHOLAR_QUERY_URL = ScholarConf.GS_SCHOLAR_SITE + '/scholar?' \
        + 'as_q=%(words)s' \
        + '&as_epq=%(phrase)s' \
        + '&as_oq=%(words_some)s' \
        + '&as_eq=%(words_none)s' \
        + '&as_occt=%(scope)s' \
        + '&as_sauthors=%(authors)s' \
        + '&as_publication=%(pub)s' \
        + '&as_ylo=%(ylo)s' \
        + '&as_yhi=%(yhi)s' \
        + '&as_sdt=%(patents)s%%2C5' \
        + '&as_vis=%(citations)s' \
        + '&btnG=&hl=en' \
        + '&num=%(num)s'
    SS_SCHOLAR_QUERY_URL = ScholarConf.SS_SCHOLAR_SITE + '/search?' \
        + 'q=%(phrase)s' \
        + '&sort=%(sortby)s' \
        + '&ae=%(ae)s' 

    def __init__(self):
        ScholarQuery.__init__(self)
        self._add_attribute_type('num_results', 'Results', 0)
        self.words = None # The default search behavior
        self.words_some = None # At least one of those words
        self.words_none = None # None of these words
        self.phrase = None
        self.scope_title = False # If True, search in title only
        self.author = None 
        self.pub = None
        self.timeframe = [None, None]
        self.include_patents = True
        self.include_citations = True
        self.sortby = 'relevance'
        self.ae = False
    

    def set_words(self, words):
        """Sets words that *all* must be found in the result."""
        self.words = words

    def set_words_some(self, words):
        """Sets words of which *at least one* must be found in result."""
        self.words_some = words

    def set_words_none(self, words):
        """Sets words of which *none* must be found in the result."""
        self.words_none = words

    def set_phrase(self, phrase):
        """Sets phrase that must be found in the result exactly."""
        self.phrase = phrase
        
    def set_sortby(self, sortby):
        """Sets phrase that must be found in the result exactly."""
        self.sortby = sortby
        
    def set_ae(self, ae):
        """Sets phrase that must be found in the result exactly."""
        self.ae = ae
    
    def set_scope(self, title_only):
        """
        Sets Boolean indicating whether to search entire article or title
        only.
        """
        self.scope_title = title_only

    def set_author(self, author):
        """Sets names that must be on the result's author list."""
        self.author = author

    def set_pub(self, pub):
        """Sets the publication in which the result must be found."""
        self.pub = pub

    def set_timeframe(self, start=None, end=None):
        """
        Sets timeframe (in years as integer) in which result must have
        appeared. It's fine to specify just start or end, or both.
        """
        if start:
            start = ScholarUtils.ensure_int(start)
        if end:
            end = ScholarUtils.ensure_int(end)
        self.timeframe = [start, end]

    def set_include_citations(self, yesorno):
        self.include_citations = yesorno

    def set_include_patents(self, yesorno):
        self.include_patents = yesorno

    def get_url(self, choice):
        if self.words is None and self.words_some is None \
           and self.words_none is None and self.phrase is None \
           and self.author is None and self.pub is None \
           and self.timeframe[0] is None and self.timeframe[1] is None:
            raise QueryArgumentError('search query needs more parameters')

        # If we have some-words or none-words lists, we need to
        # process them so GS understands them. For simple
        # space-separeted word lists, there's nothing to do. For lists
        # of phrases we have to ensure quotations around the phrases,
        # separating them by whitespace.
        words_some = None
        words_none = None

        if self.words_some:
            words_some = self._parenthesize_phrases(self.words_some)
        if self.words_none:
            words_none = self._parenthesize_phrases(self.words_none)

        gs_urlargs = {'words': self.words or '',
                   'words_some': words_some or '',
                   'words_none': words_none or '',
                   'phrase': self.phrase or '',
                   'scope': 'title' if self.scope_title else 'any',
                   'authors': self.author or '',
                   'pub': self.pub or '',
                   'ylo': self.timeframe[0] or '',
                   'yhi': self.timeframe[1] or '',
                   'patents': '0' if self.include_patents else '1',
                   'citations': '0' if self.include_citations else '1',
                   'num': self.num_results or ScholarConf.MAX_PAGE_RESULTS}
           
        ss_urlargs = {'phrase': self.phrase or '',
                   'sortby': self.sortby or '',
                   'ae': self.ae or 'false'}
        
        if choice == 'gs':
            self.SCHOLAR_QUERY_URL = self.GS_SCHOLAR_QUERY_URL
            self.urlargs = gs_urlargs
        elif choice == 'ss':
            self.SCHOLAR_QUERY_URL = self.SS_SCHOLAR_QUERY_URL
            self.urlargs = ss_urlargs  
    
        for key, val in self.urlargs.items():
            self.urlargs[key] = quote(encode(val))

        return self.SCHOLAR_QUERY_URL % self.urlargs


class ScholarSettings(object):

    """
    This class lets you adjust the Scholar settings for your
    session. It's intended to mirror the features tunable in the
    Scholar Settings pane, but right now it's a bit basic.
    """
    CITFORM_NONE = 0
    CITFORM_REFWORKS = 1
    CITFORM_REFMAN = 2
    CITFORM_ENDNOTE = 3
    CITFORM_BIBTEX = 4

    def __init__(self):
        self.citform = 0 # Citation format, default none
        self.per_page_results = ScholarConf.MAX_PAGE_RESULTS
        self._is_configured = False

    def set_citation_format(self, citform):
        citform = ScholarUtils.ensure_int(citform)
        if citform < 0 or citform > self.CITFORM_BIBTEX:
            raise FormatError('citation format invalid, is "%s"' \
                              % citform)
        self.citform = citform
        self._is_configured = True

    def set_per_page_results(self, per_page_results):
        msg = 'page results must be integer'
        self.per_page_results = ScholarUtils.ensure_int(per_page_results, msg)
        self.per_page_results = min(self.per_page_results,
                                    ScholarConf.MAX_PAGE_RESULTS)
        self._is_configured = True

    def is_configured(self):
        return self._is_configured


class ScholarQuerier(object):

    """
    ScholarQuerier instances can conduct a search on Google Scholar
    with subsequent parsing of the resulting HTML content.  The
    articles found are collected in the articles member, a list of
    ScholarArticle instances.
    """

    # Default URLs for visiting and submitting Settings pane, as of 3/14
    GET_SETTINGS_URL = ScholarConf.GS_SCHOLAR_SITE + '/scholar_settings?' \
        + 'sciifh=1&hl=en&as_sdt=0,5'

    SET_SETTINGS_URL = ScholarConf.GS_SCHOLAR_SITE + '/scholar_setprefs?' \
        + 'q=' \
        + '&scisig=%(scisig)s' \
        + '&inststart=0' \
        + '&as_sdt=1,5' \
        + '&as_sdtp=' \
        + '&num=%(num)s' \
        + '&scis=%(scis)s' \
        + '%(scisf)s' \
        + '&hl=en&lang=all&instq=&inst=569367360547434339&save='

    # Older URLs:
    # ScholarConf.SCHOLAR_SITE + '/scholar?q=%s&hl=en&btnG=Search&as_sdt=2001&as_sdtp=on

    class Parser(ScholarArticleParser120726):
        def __init__(self, querier, choice):
            ScholarArticleParser120726.__init__(self, choice)
            self.querier = querier

        def handle_num_results(self, num_results):
            if self.querier is not None and self.querier.query is not None:
                self.querier.query['num_results'] = num_results

        def handle_article(self, art):
            self.querier.add_article(art)

    def __init__(self, choice):
        self.articles = []
        self.query = None
        self.cjar = MozillaCookieJar()
        self.choice = choice

        # If we have a cookie file, load it:
        if ScholarConf.COOKIE_JAR_FILE and \
           os.path.exists(ScholarConf.COOKIE_JAR_FILE):
            try:
                self.cjar.load(ScholarConf.COOKIE_JAR_FILE,
                               ignore_discard=True)
                ScholarUtils.log('info', 'loaded cookies file')
            except Exception as msg:
                ScholarUtils.log('warn', 'could not load cookies file: %s' % msg)
                self.cjar = MozillaCookieJar() # Just to be safe

        self.opener = build_opener(HTTPCookieProcessor(self.cjar))
        self.settings = None # Last settings object, if any

    def apply_settings(self, settings):
        """
        Applies settings as provided by a ScholarSettings instance.
        """
        if settings is None or not settings.is_configured():
            return True

        self.settings = settings

        # This is a bit of work. We need to actually retrieve the
        # contents of the Settings pane HTML in order to extract
        # hidden fields before we can compose the query for updating
        # the settings.
        html = self._get_http_response(url=self.GET_SETTINGS_URL,
                                       log_msg='dump of settings form HTML',
                                       err_msg='requesting settings failed')
        if html is None:
            return False

        # Now parse the required stuff out of the form. We require the
        # "scisig" token to make the upload of our settings acceptable
        # to Google.
        soup = BeautifulSoup(html)

        tag = soup.find(name='form', attrs={'id': 'gs_settings_form'})
        if tag is None:
            ScholarUtils.log('info', 'parsing settings failed: no form')
            return False

        tag = tag.find('input', attrs={'type':'hidden', 'name':'scisig'})
        if tag is None:
            ScholarUtils.log('info', 'parsing settings failed: scisig')
            return False

        urlargs = {'scisig': tag['value'],
                   'num': settings.per_page_results,
                   'scis': 'no',
                   'scisf': ''}

        if settings.citform != 0:
            urlargs['scis'] = 'yes'
            urlargs['scisf'] = '&scisf=%d' % settings.citform

        html = self._get_http_response(url=self.SET_SETTINGS_URL % urlargs,
                                       log_msg='dump of settings result HTML',
                                       err_msg='applying setttings failed')
        if html is None:
            return False

        ScholarUtils.log('info', 'settings applied')
        return True

    def send_query(self, query):
        """
        This method initiates a search query (a ScholarQuery instance)
        with subsequent parsing of the response.
        """
        self.clear_articles()
        self.query = query
        url = query.get_url(self.choice)
        #print("URL :- ", url)
        html = self._get_http_response(url,
                                       log_msg='dump of query response HTML',
                                       err_msg='results retrieval failed')
        #print("HTML :- ", html)
        if html is None:
            return

        self.parse(html, self.choice)

    def get_citation_data(self, article):
        """
        Given an article, retrieves citation link. Note, this requires that
        you adjusted the settings to tell Google Scholar to actually
        provide this information, *prior* to retrieving the article.
        """
        if article['url_citation'] is None:
            return False
        if article.citation_data is not None:
            return True

        ScholarUtils.log('info', 'retrieving citation export data')
        data = self._get_http_response(url=article['url_citation'],
                                       log_msg='citation data response',
                                       err_msg='requesting citation data failed')
        if data is None:
            return False

        article.set_citation_data(data)
        return True

    def parse(self, html, choice):
        """
        This method allows parsing of provided HTML content.
        """
        parser = self.Parser(self, choice)
        parser.parse(html, choice)

    def add_article(self, art):
        self.get_citation_data(art)
        self.articles.append(art)

    def clear_articles(self):
        """Clears any existing articles stored from previous queries."""
        self.articles = []

    def save_cookies(self):
        """
        This stores the latest cookies we're using to disk, for reuse in a
        later session.
        """
        if ScholarConf.COOKIE_JAR_FILE is None:
            return False
        try:
            self.cjar.save(ScholarConf.COOKIE_JAR_FILE,
                           ignore_discard=True)
            ScholarUtils.log('info', 'saved cookies file')
            return True
        except Exception as msg:
            ScholarUtils.log('warn', 'could not save cookies file: %s' % msg)
            return False

    def _get_http_response(self, url, log_msg=None, err_msg=None):
        """
        Helper method, sends HTTP request and returns response payload.
        """
        if log_msg is None:
            log_msg = 'HTTP response data follow'
        if err_msg is None:
            err_msg = 'request failed'
        try:
            ScholarUtils.log('info', 'requesting %s' % unquote(url))

            req = Request(url=url, headers={'User-Agent': ScholarConf.USER_AGENT})
            hdl = self.opener.open(req)
            html = hdl.read()

            ScholarUtils.log('debug', log_msg)
            ScholarUtils.log('debug', '>>>>' + '-'*68)
            ScholarUtils.log('debug', 'url: %s' % hdl.geturl())
            ScholarUtils.log('debug', 'result: %s' % hdl.getcode())
            ScholarUtils.log('debug', 'headers:\n' + str(hdl.info()))
            ScholarUtils.log('debug', 'data:\n' + html.decode('utf-8')) # For Python 3
            ScholarUtils.log('debug', '<<<<' + '-'*68)

            return html
        except Exception as err:
            ScholarUtils.log('info', err_msg + ': %s' % err)
            return None


def txt(querier, with_globals):
    if with_globals:
        # If we have any articles, check their attribute labels to get
        # the maximum length -- makes for nicer alignment.
        max_label_len = 0
        if len(querier.articles) > 0:
            items = sorted(list(querier.articles[0].attrs.values()),
                           key=lambda item: item[2])
            max_label_len = max([len(str(item[1])) for item in items])

        # Get items sorted in specified order:
        items = sorted(list(querier.query.attrs.values()), key=lambda item: item[2])
        # Find largest label length:
        max_label_len = max([len(str(item[1])) for item in items] + [max_label_len])
        fmt = '[G] %%%ds %%s' % max(0, max_label_len-4)
        for item in items:
            if item[0] is not None:
                print(fmt % (item[1], item[0]))
        if len(items) > 0:
            print

    articles = querier.articles
    for art in articles:
        print(encode(art.as_txt()) + '\n')

def csv(querier, header=False, sep='|'):
    articles = querier.articles
    for art in articles:
        result = art.as_csv(header=header, sep=sep)
        print(encode(result))
        header = False

def citation_export(querier):
    articles = querier.articles
    for art in articles:
        print(art.as_citation() + '\n')
        
def getQueriesFromACMClassificationSystem() :
    g = Graph()
    g.parse("test.xml", format="xml")
    loader = skos.RDFLoader(g)
    concept_keys = loader.keys()
    concept_labels = []
    for i in concept_keys :
        if hasattr(loader[i], 'prefLabel'):
            concept_labels.append(loader[i].prefLabel)
    return concept_labels   

def getGranulatedQueriesFromACMClassificationSystem() :
    g = Graph()
    g.parse("test.xml", format="xml")
    loader = skos.RDFLoader(g)
    concept_keys = loader.keys()
    concept_labels = []
    master_list = []
    leaf_labels = []
    for i in concept_keys :
        if hasattr(loader[i], 'prefLabel'):
            if len(loader[i].narrower) == 0:
                leaf_labels.append(loader[i].prefLabel)
    top_labels = []
    for i in concept_keys :
        if hasattr(loader[i], 'prefLabel'):
            if len(loader[i].broader) == 0:
                top_labels.append(loader[i].prefLabel)
    second_labels = []
    for i in concept_keys :
        if hasattr(loader[i], 'prefLabel'):
            if len(loader[i].broader) == 1 and len(loader[i].narrower) >= 1:
                second_labels.append(loader[i].prefLabel)
    third_labels = []
    for i in concept_keys :
        if hasattr(loader[i], 'prefLabel'):
            if len(loader[i].broader) == 2 and len(loader[i].narrower) >= 1:
                third_labels.append(loader[i].prefLabel)
    fourth_labels = []
    for i in concept_keys :
        if hasattr(loader[i], 'prefLabel'):
            if len(loader[i].broader) == 3 and len(loader[i].narrower) >= 1:
                fourth_labels.append(loader[i].prefLabel)   
    for i in concept_keys :
        if hasattr(loader[i], 'prefLabel'):
            concept_labels.append(loader[i].prefLabel)
    master_list.append(top_labels)  
    master_list.append(second_labels)
    master_list.append(third_labels)
    master_list.append(fourth_labels)
    master_list.append(leaf_labels) 
    master_list.append(concept_labels)   
                      
    return master_list        
        
def getResponseFromScopus(query):
    query = query.replace(' ', '+')
    SCOPUS_SCHOLAR_SITE = 'http://api.elsevier.com/content'
    hdrs = {'Accept':'application/json', 'X-ELS-APIKey': MY_SOCPUS_API_KEY }
    SCOPUS_SCHOLAR_QUERY_URL = SCOPUS_SCHOLAR_SITE + '/search/scopus?' \
        + 'query=title-abs-key('+query+')&SUBJAREA(COMP)&field=title,doi,coverDate&count=10'
    resp = requests.get(url = SCOPUS_SCHOLAR_QUERY_URL, headers = hdrs)
    scopus_response = resp.json()
    #print(json.dumps(scopus_response, sort_keys=True, indent=4, separators=(',', ': ')))
    return scopus_response

def getResponseFromMAS(query) :
    driver = webdriver.Firefox()
    #iquery = '%40'+query+'%40'
    #url = 'https://academic.microsoft.com/#/search?iq='+iquery+'&q='+query+'&filters=&from=0&sort=0'
    #driver.implicitly_wait(16)
    url = 'https://academic.microsoft.com/#/search?'
    driver.get(url)
    element = driver.find_element_by_id("searchControl")
    element.send_keys(query)
    element.send_keys(Keys.ENTER)
    time.sleep(7)
    html = driver.page_source
    soup = BeautifulSoup(html)
    mytitles = soup.findAll("a", { "class" : "blue-title" })
    mas_articles = []
    if not len(mytitles) == 0 :
        for i in mytitles :
            mas_articles.append(i.text)
    driver.quit()
    return mas_articles

def getResponseFromGS(gs_driver,query) :
    #driver = webdriver.Firefox()
    query.replace(' ', '+')
    url = 'https://scholar.google.co.in/scholar?hl=en&q='+query
    gs_driver.get(url)
    #driver.implicitly_wait(16)
    wait = ui.WebDriverWait(gs_driver,25)
    html = gs_driver.page_source
    soup = BeautifulSoup(html)
    mytitles = soup.findAll("h3", { "class" : "gs_rt" })
    time.sleep(7)
    gs_articles = []
    if not len(mytitles) == 0 :
        for i in mytitles :
            atag = i.a
            if not atag is None :
                txt = ''.join(atag.findAll(text=True))
                gs_articles.append(txt)
    return gs_articles
    
def main():
    usage = """scholar.py [options] <query string>
A command-line interface to Google Scholar.

Examples:

# Retrieve one article written by Einstein on quantum theory:
scholar.py -c 1 --author "albert einstein" --phrase "quantum theory"

# Retrieve a BibTeX entry for that quantum theory paper:
scholar.py -c 1 -C 17749203648027613321 --citation bt

# Retrieve five articles written by Einstein after 1970 where the title
# does not contain the words "quantum" and "theory":
scholar.py -c 5 -a "albert einstein" -t --none "quantum theory" --after 1970"""

    fmt = optparse.IndentedHelpFormatter(max_help_position=50, width=100)
    parser = optparse.OptionParser(usage=usage, formatter=fmt)
    group = optparse.OptionGroup(parser, 'Query arguments',
                                 'These options define search query arguments and parameters.')
    group.add_option('-a', '--author', metavar='AUTHORS', default=None,
                     help='Author name(s)')
    group.add_option('-A', '--all', metavar='WORDS', default=None, dest='allw',
                     help='Results must contain all of these words')
    group.add_option('-s', '--some', metavar='WORDS', default=None,
                     help='Results must contain at least one of these words. Pass arguments in form -s "foo bar baz" for simple words, and -s "a phrase, another phrase" for phrases')
    group.add_option('-n', '--none', metavar='WORDS', default=None,
                     help='Results must contain none of these words. See -s|--some re. formatting')
    group.add_option('-p', '--phrase', metavar='PHRASE', default=None,
                     help='Results must contain exact phrase')
    group.add_option('-t', '--title-only', action='store_true', default=False,
                     help='Search title only')
    group.add_option('-P', '--pub', metavar='PUBLICATIONS', default=None,
                     help='Results must have appeared in this publication')
    group.add_option('--after', metavar='YEAR', default=None,
                     help='Results must have appeared in or after given year')
    group.add_option('--before', metavar='YEAR', default=None,
                     help='Results must have appeared in or before given year')
    group.add_option('--no-patents', action='store_true', default=False,
                     help='Do not include patents in results')
    group.add_option('--no-citations', action='store_true', default=False,
                     help='Do not include citations in results')
    group.add_option('-C', '--cluster-id', metavar='CLUSTER_ID', default=None,
                     help='Do not search, just use articles in given cluster ID')
    group.add_option('-c', '--count', type='int', default=None,
                     help='Maximum number of results')
    parser.add_option_group(group)

    group = optparse.OptionGroup(parser, 'Output format',
                                 'These options control the appearance of the results.')
    group.add_option('--txt', action='store_true',
                     help='Print article data in text format (default)')
    group.add_option('--txt-globals', action='store_true',
                     help='Like --txt, but first print global results too')
    group.add_option('--csv', action='store_true',
                     help='Print article data in CSV form (separator is "|")')
    group.add_option('--csv-header', action='store_true',
                     help='Like --csv, but print header with column names')
    group.add_option('--citation', metavar='FORMAT', default=None,
                     help='Print article details in standard citation format. Argument Must be one of "bt" (BibTeX), "en" (EndNote), "rm" (RefMan), or "rw" (RefWorks).')
    parser.add_option_group(group)

    group = optparse.OptionGroup(parser, 'Miscellaneous')
    group.add_option('--cookie-file', metavar='FILE', default=None,
                     help='File to use for cookie storage. If given, will read any existing cookies if found at startup, and save resulting cookies in the end.')
    group.add_option('-d', '--debug', action='count', default=0,
                     help='Enable verbose logging to stderr. Repeated options increase detail of debug output.')
    group.add_option('-v', '--version', action='store_true', default=False,
                     help='Show version information')
    parser.add_option_group(group)

    options, _ = parser.parse_args()

    # Show help if we have neither keyword search nor author name
    if len(sys.argv) == 1:
        parser.print_help()
        return 1

    if options.debug > 0:
        options.debug = min(options.debug, ScholarUtils.LOG_LEVELS['debug'])
        ScholarConf.LOG_LEVEL = options.debug
        ScholarUtils.log('info', 'using log level %d' % ScholarConf.LOG_LEVEL)

    if options.version:
        print('This is scholar.py %s.' % ScholarConf.VERSION)
        return 0

    if options.cookie_file:
        ScholarConf.COOKIE_JAR_FILE = options.cookie_file

    # Sanity-check the options: if they include a cluster ID query, it
    # makes no sense to have search arguments:
    if options.cluster_id is not None:
        if options.author or options.allw or options.some or options.none \
           or options.phrase or options.title_only or options.pub \
           or options.after or options.before:
            print('Cluster ID queries do not allow additional search arguments.')
            return 1

    
    #concept_labels = getQueriesFromACMClassificationSystem()
    master_list = getGranulatedQueriesFromACMClassificationSystem()
    top_sum_score = {'GS_SS' : 0.0, 'SS_SCOPUS' : 0.0, 'SCOPUS_MAS': 0.0, 'MAS_GS' : 0.0, 'GS_SCOPUS' : 0.0, 'SS_MAS' : 0.0, 'GS_SS_SCOPUS' :0.0, 'SS_SCOPUS_MAS' :0.0, 'SCOPUS_MAS_GS' :0.0, 'MAS_GS_SS' :0.0, 'GS_SS_SCOPUS_MAS' :0.0}
    second_sum_score = {'GS_SS' : 0.0, 'SS_SCOPUS' : 0.0, 'SCOPUS_MAS': 0.0, 'MAS_GS' : 0.0, 'GS_SCOPUS' : 0.0, 'SS_MAS' : 0.0, 'GS_SS_SCOPUS' :0.0, 'SS_SCOPUS_MAS' :0.0, 'SCOPUS_MAS_GS' :0.0, 'MAS_GS_SS' :0.0, 'GS_SS_SCOPUS_MAS' :0.0}
    third_sum_score = {'GS_SS' : 0.0, 'SS_SCOPUS' : 0.0, 'SCOPUS_MAS': 0.0, 'MAS_GS' : 0.0, 'GS_SCOPUS' : 0.0, 'SS_MAS' : 0.0, 'GS_SS_SCOPUS' :0.0, 'SS_SCOPUS_MAS' :0.0, 'SCOPUS_MAS_GS' :0.0, 'MAS_GS_SS' :0.0, 'GS_SS_SCOPUS_MAS' :0.0}
    fourth_sum_score = {'GS_SS' : 0.0, 'SS_SCOPUS' : 0.0, 'SCOPUS_MAS': 0.0, 'MAS_GS' : 0.0, 'GS_SCOPUS' : 0.0, 'SS_MAS' : 0.0, 'GS_SS_SCOPUS' :0.0, 'SS_SCOPUS_MAS' :0.0, 'SCOPUS_MAS_GS' :0.0, 'MAS_GS_SS' :0.0, 'GS_SS_SCOPUS_MAS' :0.0}
    leaf_sum_score = {'GS_SS' : 0.0, 'SS_SCOPUS' : 0.0, 'SCOPUS_MAS': 0.0, 'MAS_GS' : 0.0, 'GS_SCOPUS' : 0.0, 'SS_MAS' : 0.0, 'GS_SS_SCOPUS' :0.0, 'SS_SCOPUS_MAS' :0.0, 'SCOPUS_MAS_GS' :0.0, 'MAS_GS_SS' :0.0, 'GS_SS_SCOPUS_MAS' :0.0}
    sum_score = {'GS_SS' : 0.0, 'SS_SCOPUS' : 0.0, 'SCOPUS_MAS': 0.0, 'MAS_GS' : 0.0, 'GS_SCOPUS' : 0.0, 'SS_MAS' : 0.0, 'GS_SS_SCOPUS' :0.0, 'SS_SCOPUS_MAS' :0.0, 'SCOPUS_MAS_GS' :0.0, 'MAS_GS_SS' :0.0, 'GS_SS_SCOPUS_MAS' :0.0}
    gs_driver = webdriver.Firefox()
    cnt = 0
    """concept_labels = []
    with open('Input/popular_tags.txt') as f:
        content = f.readlines()
    for m in content :
        m = m.rstrip()
        m = unicode(m).encode("utf-8")
        concept_labels.append(m)"""
        
    for i in range(0,len(master_list[4])) :
        cl = master_list[4][i]
        cl = cl.rstrip()
        cl = unicode(cl).encode("utf-8")
        print("Running experiment for topic : %s" % cl)
        #querier_gs = ScholarQuerier('gs')
        querier_ss = ScholarQuerier('ss')
        #settings_gs = ScholarSettings()
        settings_ss = ScholarSettings()
        querier_scopus = getResponseFromScopus(cl)
        querier_mas = getResponseFromMAS(cl)
        new_querier_gs = getResponseFromGS(gs_driver,cl)
    
        if options.citation == 'bt':
            #settings_gs.set_citation_format(ScholarSettings.CITFORM_BIBTEX)
            settings_ss.set_citation_format(ScholarSettings.CITFORM_BIBTEX)
        elif options.citation == 'en':
            #settings_gs.set_citation_format(ScholarSettings.CITFORM_ENDNOTE)
            settings_ss.set_citation_format(ScholarSettings.CITFORM_BIBTEX)
        elif options.citation == 'rm':
            #settings_gs.set_citation_format(ScholarSettings.CITFORM_REFMAN)
            settings_ss.set_citation_format(ScholarSettings.CITFORM_BIBTEX)
        elif options.citation == 'rw':
            #settings_gs.set_citation_format(ScholarSettings.CITFORM_REFWORKS)
            settings_ss.set_citation_format(ScholarSettings.CITFORM_BIBTEX)
        elif options.citation is not None:
            print('Invalid citation link format, must be one of "bt", "en", "rm", or "rw".')
            return 1
    
        #querier_gs.apply_settings(settings_gs)
    
        if options.cluster_id:
            print('') 
        else:
            query = SearchScholarQuery()
            if options.author:
                query.set_author(options.author)
            if options.allw:
                query.set_words(options.allw)
            if options.some:
                query.set_words_some(options.some)
            if options.none:
                query.set_words_none(options.none)
            if options.phrase:
                query.set_phrase(cl)
            if options.title_only:
                query.set_scope(True)
            if options.pub:
                query.set_pub(options.pub)
            if options.after or options.before:
                query.set_timeframe(options.after, options.before)
            if options.no_patents:
                query.set_include_patents(False)
            if options.no_citations:
                query.set_include_citations(False)
    
        if options.count is not None:
            options.count = min(options.count, ScholarConf.MAX_PAGE_RESULTS)
            query.set_num_page_results(options.count)
    
        #querier_gs.send_query(query)
        #time.sleep(2)
        querier_ss.send_query(query)
        time.sleep(2)
       
        if options.csv:
            #csv(querier_gs)
            csv(querier_ss)
        elif options.csv_header:
            #csv(querier_gs, header=True)
            csv(querier_ss, header=True)
        elif options.citation is not None:
            #citation_export(querier_gs)
            citation_export(querier_ss)
        """else:
            txt(querier_gs, with_globals=options.txt_globals)
            txt(querier_ss, with_globals=options.txt_globals)"""
        if options.cookie_file:
            #querier_gs.save_cookies()
            querier_ss.save_cookies()
        #gs_articles = [gg.__getitem__('title') for gg in querier_gs.articles]
        ss_articles = [ss.__getitem__('title') for ss in querier_ss.articles]
        #print('len-->', len(ss_articles))
        mas_articles = querier_mas
        gs_articles = new_querier_gs
        # To combat None results
        check = 0
        scopus_articles = []
        if 'search-results' in querier_scopus.keys() :
            if 'entry' in querier_scopus['search-results'].keys() :
                for ss in querier_scopus['search-results']['entry'] :
                    if 'dc:title' in ss.keys() :
                        scopus_articles.append(ss['dc:title'])
                #scopus_articles = [ss['dc:title'] for ss in querier_scopus['search-results']['entry']]
                    else :
                        check = 1
                        print('check changed in scopus')
            else :
                check = 1
                print('check changed in scopus')
        else :
            check = 1
            print('check changed in scopus')
        for ss in gs_articles :
            if ss is None :
                check = 1
                print('check changed in gs_articles')
        for ss in ss_articles :
            if ss is None :
                check = 1
                print('check changed in ss_articles')
        
        print('+++++++++++++++++++++++++++++++++++++')
        for ab in gs_articles :
            print('gs_articles --> ',ab)
        print('+++++++++++++++++++++++++++++++++++++')
        for ab in ss_articles :
            print('ss_articles --> ',ab)
        print('+++++++++++++++++++++++++++++++++++++')
        for ab in scopus_articles :
            print('scopus_articles --> ',ab)
        print('+++++++++++++++++++++++++++++++++++++')
        for ab in mas_articles :
            print('mas_articles --> ',ab)
        
        if check == 1 :
            continue
        # Setting size  of result-sets between minimum 0 to maximum 8
        gs_articles = gs_articles[:8] if len(gs_articles)>8 else gs_articles
        ss_articles = ss_articles[:8] if len(ss_articles)>8 else ss_articles
        mas_articles = mas_articles[:8] if len(mas_articles)>8 else mas_articles
        scopus_articles = scopus_articles[:8] if len(scopus_articles)>8 else scopus_articles
        gs_articles = set(gs_articles)
        ss_articles = set(ss_articles)
        mas_articles = set(mas_articles)
        scopus_articles = set(scopus_articles)
        if len(gs_articles) > 0 and len(ss_articles) > 0 :
            score = len(set.intersection(gs_articles,ss_articles))/float(len(set.union(gs_articles,ss_articles)))
            sum_score['GS_SS'] += score
        else :
            continue
        if len(scopus_articles) > 0 and len(ss_articles) > 0 :
            score = len(set.intersection(ss_articles,scopus_articles))/float(len(set.union(ss_articles,scopus_articles)))
            sum_score['SS_SCOPUS'] += score
        else :
            continue
        if len(mas_articles) > 0 and len(scopus_articles) > 0 :
            score = len(set.intersection(mas_articles,scopus_articles))/float(len(set.union(mas_articles,scopus_articles)))
            sum_score['SCOPUS_MAS'] += score
        else :
            continue
        if len(gs_articles) > 0 and len(mas_articles) > 0 :
            score = len(set.intersection(mas_articles,gs_articles))/float(len(set.union(mas_articles,gs_articles)))
            sum_score['MAS_GS'] += score
        else :
            continue
        if len(gs_articles) > 0 and len(scopus_articles) > 0 :
            score = len(set.intersection(scopus_articles,gs_articles))/float(len(set.union(scopus_articles,gs_articles)))
            sum_score['GS_SCOPUS'] += score
        else :
            continue
        if len(mas_articles) > 0 and len(ss_articles) > 0 :
            score = len(set.intersection(mas_articles,ss_articles))/float(len(set.union(mas_articles,ss_articles)))
            sum_score['SS_MAS'] += score
        else :
            continue
        if len(gs_articles) > 0 and len(ss_articles) > 0 and len(scopus_articles) > 0 :
            score = len(set.intersection(ss_articles,gs_articles,scopus_articles))/float(len(set.union(ss_articles,gs_articles,scopus_articles)))
            sum_score['GS_SS_SCOPUS'] += score
        else :
            continue
        if len(mas_articles) > 0 and len(ss_articles) > 0 and len(scopus_articles) > 0 :
            score = len(set.intersection(ss_articles,mas_articles,scopus_articles))/float(len(set.union(ss_articles,mas_articles,scopus_articles)))
            sum_score['SS_SCOPUS_MAS'] += score
        else :
            continue
        if len(mas_articles) > 0 and len(gs_articles) > 0 and len(scopus_articles) > 0 :
            score = len(set.intersection(mas_articles,gs_articles,scopus_articles))/float(len(set.union(mas_articles,gs_articles,scopus_articles)))
            sum_score['SCOPUS_MAS_GS'] += score
        else :
            continue
        if len(gs_articles) > 0 and len(ss_articles) > 0 and len(mas_articles) > 0 :
            score = len(set.intersection(mas_articles,gs_articles,ss_articles))/float(len(set.union(mas_articles,gs_articles,ss_articles)))
            sum_score['MAS_GS_SS'] += score
        else :
            continue
        if len(gs_articles) > 0 and len(ss_articles) > 0  and len(mas_articles) > 0 and len(scopus_articles) > 0 :
            score = len(set.intersection(mas_articles,gs_articles,ss_articles,scopus_articles))/float(len(set.union(mas_articles,gs_articles,ss_articles,scopus_articles)))
            sum_score['GS_SS_SCOPUS_MAS'] += score
        else :
            continue
        
        for k in sum_score.keys():
            print('Key %s ---> sum_score %f' % (k,sum_score[k]))
        cnt += 1
        #if(cnt >= 20):
            #break
        print('iteration ----> \n', cnt)
        print()
        print()
    #avg_score = sum_score / float(len(concept_labels))
    for k in sum_score.keys():
        sum_score[k] = sum_score[k] / float(cnt)
        #sum_score[k] = sum_score[k] / float(len(concept_labels))
        print('Key %s ---> avg_sum_score %f' % (k,sum_score[k]))
    gs_driver.quit()
    return 0

if __name__ == "__main__":
    sys.exit(main())
