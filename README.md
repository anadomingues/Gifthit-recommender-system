# GiftHit - A gift recommender based on Pinterest profiles

This project was built in 2 weeks at the final stage of the Galvanize Data Science Immersive program.

## Motivation

Every year at the holiday season, we always find the same problem over and over again: What should I give to my cousin? How many times we find ourselves having no idea what to get someone for their birthday?

A great gift (Gift hit) is the one where you show that you know who the person is and what he/she is about.

The idea of this project is to recommend a gift based on people's Pinterest pins. Pinterest is an application that allows us to save pins we like by topic.
By knowing the personal taste of each user of Pinterest we can then recommend something that this person has a high interest on. Based on Amazon's gift Finder section the app will then recommend a gift that he/she will have a high probability to like.

The work will be presented by a web application in a near future.

## Data

### Pinterest Profiles

In order to obtain user information and their pins, I scrape Pinterest usernames first by extracting random usernames and after their followers names. I extract a total of 50000 users with respective pins information.
The information from the pins is obtained by an RSS feed: https://www.pinterest.com/username/feed.rss

### Products

The products from the recommendation system were scrapes from Amazon.com Gift Finder Section.

![Pins](images/pins.png|width=100)
