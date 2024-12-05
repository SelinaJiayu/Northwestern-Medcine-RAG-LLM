import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import argparse 
import os


def find_topics_url(base_url):
    response = requests.get(base_url)
    html_content = response.content

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all div elements with class "col-sm-6"
    col_sm_6_divs = soup.find_all('div', class_='col-sm-6')

    # List to store the links
    links = []

    # Loop through each "col-sm-6" div to find anchor tags and extract their URLs
    for div in col_sm_6_divs:
        anchor_tag = div.find('a', href=True)
        if anchor_tag:
            href = anchor_tag['href']
            link_text = anchor_tag.get_text(strip=True)
            links.append((href, link_text))
    return links 


def scrape_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract all the headings (h2, h3) and corresponding <ul> lists and <p> tags below them
    content = []

    # Extract sections based on headings (h2, h3)
    headings = soup.find_all(['h2'])#, 'h3'])
    for heading in headings:
        title = heading.get_text(strip=True)
        section_content = {
            'title': title,
            'details': []
        }
        for sibling in heading.find_next_siblings():
            section_content['details'].append(sibling.get_text())
        content.append(section_content)
    return content

def filter_extracted_data(data):
    final_data = []
    for entry in data: 
        if len(entry.get('details')) == 0:
            continue
        else:
            entry['details'] = '\n'.join(entry['details'])
        final_data.append(entry)
    return final_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', type=str, help='Path to save the scrapped file')

    args = parser.parse_args()
    save_path = args.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    base_url = "https://www.radiologyinfo.org/en/onco"
    topic_urls = find_topics_url(base_url)
    final_df = pd.DataFrame(columns=["topic_name", "Question", "Answer"])
    for topic_url, _ in topic_urls:
        topic_name = topic_url.split('/')[-1]
        print(f"Topic Name : {topic_name} -- Topic Url : {topic_url}")
        extracted_data = scrape_content(topic_url)
        extracted_data = filter_extracted_data(extracted_data)

        topic_df = pd.DataFrame(extracted_data).rename({"title": "Question", "details" : "Answer"}, axis=1)
        topic_df['topic_name'] = [topic_name] * len(topic_df)
        topic_df = topic_df[['topic_name', 'Question', 'Answer']]
        final_df = pd.concat([final_df, topic_df])

    final_df = final_df[(final_df.Question != "Send us your feedback") & (final_df.Question != "Additional Information and Resources")]
    print("Total Entries:", final_df.shape[0])
    print("Saving File to ", save_path)
    final_df.to_csv(save_path, index=False)  # save to csv file