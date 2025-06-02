import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = 'http://quotes.toscrape.com/'

response = requests.get(URL)

# check if the request was successful
if response.status_code == 200:
    print('Request successful!')
else:
    print('Failed to retrieve the webpage')

soup = BeautifulSoup(response.content, 'html.parser')

# print the title of the webpage to verify
quote_blocks = soup.find_all('div', class_='quote')

data = []
for quote in quote_blocks:
    quote_text = quote.find('span', class_='text').get_text(strip=True)
    author = quote.find('small', class_='author').get_text(strip=True)
    
    # append author and quote data to row list
    data.append([author, quote_text])


# convert data to a data frame and set column names
df = pd.DataFrame(data, columns=['Author', 'Quote'])

print(df)
# save to csv file
df.to_csv('scraped_data.csv',index=False )