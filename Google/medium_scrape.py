from googleapiclient.discovery import build
import json
import pickle

# Query the json from Google Search API
my_api_key = 
my_cse_id = 

def google_search(search_term, api_key, cse_id, start, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, start=start, **kwargs).execute()
    return res


def scrape_google(search_term):
    start = 1
    # Create a list object to hold links
    lst = []

    for i in range(10):
        x = google_search(search_term, my_api_key, my_cse_id, start)
        start += 10
        # Parsing the json file, put links in lst
        for p in x['items']:
          lst.append(p['link'])

    print("Number of links collected: ", str(len(lst)))

    # Pickle the list
    file_Name = "./Google/Links/" + search_term + " medium"
    # open the file for writing
    picklefile = open(file_Name,'wb')
    # this writes the object a to the
    pickle.dump(lst, picklefile)
    # here we close the fileObject
    picklefile.close()
