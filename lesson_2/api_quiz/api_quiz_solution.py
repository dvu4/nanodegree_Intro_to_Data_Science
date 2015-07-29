import json
import requests

import json
import requests

def api_get_request(url):
    # In this exercise, you want to call the last.fm API to get a list of the
    # top artists in Spain.
    #
    # Once you've done this, return the name of the number 1 top artist in Spain.
    data = request.get(url).text
    data = json.loads(data)
    print type(data)
    print data
    print data['topartists']['artist'][0]['name']
    # return the top artist in Spain
    return  data['topartists']['artist'][0]['name'] 


if __name__ == '__main__':
	# url should be the url to the last.fm api call which
	# will return find the top artists in Spain

	#url = # fill this in
	url = "https://ws.audioscrobbler.com/2.0/?method=album.getinfo&api_key=[API_KEY]&artist=Rihanna&album=Loud&format=json"
    print api_get_request(url) 

