# anti-fake
TAVhacks finalist. ML-driven whatsapp that uses stance detection to determine the reputability of a claim. Users can simply forward the message to the bot and it will generate a response.

![alt text](https://github.com/gabrielsaruhashi/anti-fake/blob/master/botaction.gif "Bot in action")

## Inspiration
In recent times, we’ve witnessed how fake news has become a viral targeted strategy deployed to sway and divide citizens politically, particularly in messaging platforms where fake news can easily be forwarded and shared to large groups. 
For example, in the 2018 Brazil elections, 6/10 Brazilian voters who voted for current president Jair Bolsonaro (nearly 120 Million citizens) informed themselves about the election primarily via WhatsApp and group messaging options, and nearly 3/10 voters of the messages they received in these groups contained misinformation. 

For the most part, once these fake news goes shared, it goes unchallenged. The problem with that is that misinformation is often not harmless. In India, 50 lynchings across the country that were blamed on incendiary messages spread using the app.
In India, this fake news problem has been even more severe with the upcoming 2019 elections, with 6x larger group messaging userbase. 50 people have already been lynched because of false news being shared on the messaging platforms like Whatsapp. 

Why? There has been no way to easily cross reference or fight the spreading of fake news on any platform natively, until NOW.

## The insight
The forward dilemma has been a tremendous challenge for Whatsapp. As the Guardian’s article from this week clearly illustrates, “WhatsApp’s message-forwarding mechanics have been blamed for helping the spread of fake news in part because of the way the app displays forwarded messages. A text message that has been forwarded to a new recipient is marked as forwarded in light grey text, but otherwise appears indistinguishable from an original message sent by a contact. Critics say the design ‘strips away the identity of the sender and allows messages to spread virally with little accountability’”.

Our main goal is to empower the average citizen with a tool to fight and combat the spreading of fake news by contextifying claims. Ultimately, our goal is to change a user’s habit from merely forwarding information to easily fact checking claims before sharing them. 
If a user is able to even contextify a claim before sharing it, we can significantly reduce the viral nature of fake news, preventing harm from occuring to citizens.


## The solution
We built a native AI chatbot for WhatsApp and Google Assistant that identifies/shares relevant sources to contextify to a user’s “argument” or possibly “fake news claim” and seeks to report how relevant those claims really are to the sources. We have built and integrated the first truly native experience for fact-checking between those platforms. All a user must do is forward (copy) their claim to either our WhatsApp bot or Google Assistant app, and instantly the bot returns “claim and source corroboration report/score” along with a list of relevant sources to contextify a claim.

## Engineering (author: @gabrielsaruhashi)
 
### Webscraping
**Input**
Claim (“Donald trump secured wall funding”)

**Tokenization**
[“Donald”, “trump”, “secured”, “wall”, “funding”]

**Output**
Calls EventRegistry API and returns 500 articles related to the keywords, sorted by relevance


### Reputation
**Input**
Predicted stances and list of news sources

**Update rule**
For each new source, if the source agrees with other reputable news sources, uprank it. Otherwise, downrank it

**Output**
Expanded table


### Stance-detection
We initially used UCL's MLP for FNC-1. I then iterated the model a bit, adding an extra hidden layer and the accuracy was just as good, but qualitatively, it looked like the results for the  

## Citation
@article{riedel2017fnc,
    author = {Benjamin~Riedel and Isabelle~Augenstein and Georgios~P.~Spithourakis and Sebastian~Riedel},
    title = {A simple but tough-to-beat baseline for the {Fake News Challenge} stance detection task},
    journal = {CoRR},
    volume = {abs/1707.03264},
    year = {2017},
    url = {http://arxiv.org/abs/1707.03264}
}

## Acknowledgements
HackMIT's Fake Bananas 
