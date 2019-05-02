#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Contains the parsing function for PubMed xml files
__author__ = """Giovanni Colavizza"""

from datetime import datetime, date
from bs4 import BeautifulSoup as bs
import multiprocessing, math, re, logging, codecs

logging.basicConfig(filename='logs/parser_function.log',filemode="w+",level=logging.WARNING)
logger = logging.getLogger("Secondary")

def mp_article_parser(filenames, list_of_journals, nprocs=multiprocessing.cpu_count()-1):

	def lookup_articles(filenames_chunk, out_q):

		local_storage = list()

		for f in filenames_chunk:
			# parse the article
			try:
				soup = bs(codecs.open(f, encoding="utf-8"),"html.parser")
			except UnicodeDecodeError:
				logger.warning("Unicode Error: " + f)
				continue
			# get the article meta header
			article_meta = soup.find("article-meta")
			if not article_meta:
				logger.warning("No meta nor ID: " + f)
				continue
			# get the article IDs.
			article_ids_found = article_meta.find_all("article-id")
			if len(article_ids_found) == 0:
				logger.warning("No meta nor ID: " + f)
				continue
			article_ids = list()
			id_pmc = ""
			id_pmid = ""
			id_publisher = ""
			id_doi = ""
			for article_id in article_ids_found:
				if article_id.has_attr("pub-id-type"):
					if article_id["pub-id-type"].strip() == "pmc":
						id_pmc = int(article_id.text.strip())
					elif article_id["pub-id-type"].strip() == "pmid":
						id_pmid = int(article_id.text.strip())
					elif article_id["pub-id-type"].strip() == "publisher-id":
						id_publisher = article_id.text.strip()
					elif article_id["pub-id-type"].strip() == "doi":
						id_doi = article_id.text.strip()
					article_ids.append({"id":article_id.text.strip(),"type":article_id["pub-id-type"].strip()})
			if len(article_ids) == 0:
				logger.warning("No meta nor ID: " + f)
				continue
			publication_date = datetime.now().date()
			# get the oldest available publication date
			pub_dates = article_meta.find_all("pub-date")
			if len(pub_dates) > 0:
				for pd in pub_dates:
					if pd.month:
						if pd.day:
							current_date = date(int(pd.year.text.strip()),int(pd.month.text.strip()),int(pd.day.text.strip()))
						else:
							current_date = date(int(pd.year.text.strip()), int(pd.month.text.strip()), 1)
					else:
						current_date = date(int(pd.year.text.strip()), 7, 1)
					if publication_date > current_date:
						publication_date = current_date
			else:
				logger.info("No publication date: " + f)
				continue
			if publication_date >= date(2019, 1, 1):
				logger.info("Published later than 2018 included: " + f)
				continue
			# establish is BMC or PLoS
			# First, get the journal abbreviation from the filename
			file_journal = f.split("/")[-2]
			# Initialize das_required and encouraged
			das_required = False
			das_encouraged = False
			is_plos = False
			is_bmc = False
			if file_journal in list_of_journals.keys():
				is_plos = list_of_journals[file_journal]["is_plos"]
				if not is_plos:
					is_bmc = True
				if publication_date:
					das_required = publication_date > list_of_journals[file_journal]["das_required"]
					das_encouraged = publication_date > list_of_journals[file_journal]["das_encouraged"]
			# try to get title and authors
			title = ""
			if article_meta.find("article-title"):
				title = article_meta.find("article-title").text.strip()
			else:
				logger.info("No title: " + f)
			authors = list()
			if article_meta.find("contrib-group"):
				for person in article_meta.find("contrib-group").find_all("contrib"):
					if person.has_attr('contrib-type') and person["contrib-type"] == "author":
						if person.find("surname"):
							if person.find("given-names"):
								authors.append({"name":person.find("given-names").text.strip(),"surname":person.find("surname").text.strip()})
							else:
								authors.append({"name": "", "surname": person.find("surname").text.strip()})
			else:
				logger.info("No authors: " + f)
			# get subjects (which includes type of publication, but impossible to disentangle uniformly due to subjective usage of the xml element)
			subjects = list()
			if article_meta.find("subj-group"):
				if article_meta.find("subj-group").find("subject"):
					ss = article_meta.find_all("subject")
					if len(ss) > 0:
						for r in ss:
							subjects.append(r.text.strip().lower())
			# get keywords
			keywords = list()
			if len(article_meta.find_all("kwd")) > 0:
				keywords = [x.text.strip().lower() for x in article_meta.find_all("kwd")]
			# journal name and issnS
			journal = ""
			if soup.find("journal-meta").find("journal-title"):
				journal = soup.find("journal-meta").find("journal-title").text.strip()
			journal_issns = list()
			for issn in soup.find("journal-meta").find_all("issn"):
				journal_issns.append(issn.text.strip())
			journal_issns = list(set(journal_issns))
			# abstract
			abstract = ""
			if article_meta.find("abstract"):
				abstract = article_meta.find("abstract").text.strip()

			# refs contains all the references of the given article
			refs = dict()
			# references are in the back of the article, skip if absent
			if not soup.article.back:
				logger.info("NO back for "+f)
				continue
			# skip if there is no reference list, or if it is empty, e.g. for editorials
			ref_list = soup.article.back.find("ref-list")
			if not ref_list or len(ref_list) == 0:
				logger.info("NO ref list for " + f)
				continue
			# parse each extracted reference
			n_ref = 0
			for ref in ref_list.find_all("ref"):
				# we only keep references to something that has a valid ID (DOI, pmid, pmc)
				n_ref += 1 # counts total number of references
				# get a label if it is there
				label = -1
				if ref.label:
					try:
						label = int("".join(re.findall(r'\d+', ref.label.text.strip()))) # Remove all non digits. NB it stays a string!
					except:
						label = -1
				citation_contents = ""
				if ref.find("citation"):
					citation_contents = ref.find("citation")
				elif ref.find("mixed-citation"):
					citation_contents = ref.find("mixed-citation")
				elif ref.find("element-citation"):
					citation_contents = ref.find("element-citation")
				else:
					logger.info("NO citation contents in %s, label %d"%(f,label))
					continue
				# find citation and publication identifiers
				try:
					ref_id = ref["id"]
				except:
					logger.info("NO ID for {0} in ".format(label) + f)
					continue
				# get DOI from other elements
				doi_text = ""
				if citation_contents.find("ext-link"):
					doi_text = citation_contents.find("ext-link").text.strip()
					if "doi" in doi_text or "DOI" in doi_text:
						doi_text = doi_text[doi_text.find("10."):]
					else:
						doi_text = ""
				identifiers = list()
				if len(citation_contents.find_all("pub-id")) == 0 and len(doi_text) == 0:
					logger.info("NO IDs for {0} in ".format(ref_id) + f)
					continue
				is_doi_in = False
				for identifier in citation_contents.find_all("pub-id"):
					if not identifier.has_attr('pub-id-type'):
						continue
					id_type = identifier["pub-id-type"].strip().lower()
					if id_type == "doi":
						is_doi_in = True
					identifiers.append({"id": identifier.text.strip(), "type": id_type})
				if not is_doi_in and len(doi_text) > 0:
					identifiers.append({"id": doi_text, "type": "doi"})
				# get the title
				ref_title = ""
				if citation_contents.find("article-title"):
					ref_title = citation_contents.find("article-title").text.strip()
				# get a publication year
				year = ""
				if citation_contents.find("year"):
					year = citation_contents.find("year").text.strip()
					r = re.findall(r'\d{4}', year)
					if len(r):
						year = r[0]
				try:
					year = int(year)
				except:
					logger.info("NO proper year for {0} in : {1}".format(ref_id,year) + f)
				# get a first author
				ref_authors = list()
				ref_first_author = ""
				try:
					if not citation_contents.find_all("surname"):
						logger.info("No author surnames in {1}: {0}".format(ref_id, f))
					else:
						ref_authors = [{"name": x.find("given-names").text.strip(),
										"surname": x.find("surname").text.strip()} for x in citation_contents.find_all("name")]
						ref_first_author = ref_authors[0]["surname"]
				except:
					logger.info("NO authors for {0}, {1} in ".format(ref_id,citation_contents.find("person-group")) + f)
				refs[ref_id] = {"identifiers":identifiers,"first_author":ref_first_author,"title":ref_title,"year":year,"authors":ref_authors,"label":label,"ref_id":ref_id}

			# GET DAS
			# currently only includes PLoS and BMC

			das = ""
			has_das = False

			# RULES FOR PLOS
			meta_names = soup.article.find_all("meta-name")
			for mn in meta_names:
				if "data" in mn.text.strip().lower() and "availability" in mn.text.strip().lower():
					temp_das = mn.parent.find("meta-value").text.strip()
					if temp_das:
						das = temp_das
						has_das = True
						break

			# RULES FOR BMC
			sections = soup.article.find_all("sec")
			for s in sections:
				t = s.find("title")
				if t and "data" in t.text.strip().lower() and (("availability" in t.text.strip().lower()) or ("availability" in t.text.strip().lower() and "materials" in t.text.strip().lower()) or ("accessibility" in t.text.strip().lower()) or ("sharing" in t.text.strip().lower())): # and "supporting" in t.text.lower()
					temp_das = " ".join([p.text.strip() for p in s.find_all("p")])
					if temp_das:
						das = temp_das
						has_das = True

			# create an entry for the parsed article in local_storage, that will be updated with extracted contexts
			local_storage.append({"title": title, "authors": authors, "n_authors": len(authors), "identifiers": article_ids, "n_references": n_ref,
									"id_pmc": id_pmc,
									"id_pmid": id_pmid,
									"id_publisher": id_publisher,
									"id_doi": id_doi,
									"is_plos": is_plos,
									"is_bmc": is_bmc,
									"publication_date": str(publication_date),
									"das": das,
									"has_das": has_das,
									"references": sorted([x for x in refs.values()],key=lambda x:x["label"],reverse=False),
									"keywords": keywords, "subjects": subjects, "journal": journal,
									"journal_issn": journal_issns, "filename": f, "abstract": abstract, "last_update": datetime.now(),
			                        "das_encouraged": das_encouraged, "das_required": das_required})
			logger.debug("Done: "+str(article_ids))
		out_q.put(local_storage)

	# each process will get 'chunksize' files and a queue to put stuff out
	out_q = multiprocessing.Queue()
	chunksize = int(math.ceil(len(filenames) / float(nprocs)))
	procs = []

	for i in range(nprocs):
		p = multiprocessing.Process(
			target=lookup_articles,
			args=(filenames[chunksize * i:chunksize * (i + 1)],
				  out_q))
		procs.append(p)
		p.start()

	# collect all results into a single result dict. We know how many dicts with results to expect.
	results = list()
	for _ in range(nprocs):
		results.extend(out_q.get())

	# wait for all worker processes to finish
	for p in procs:
		p.join()

	out_q.close()

	return results