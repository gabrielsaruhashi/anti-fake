# import flask dependencies
from flask import Flask
from flask import jsonify, request, make_response
# initialize the flask app
app = Flask(__name__)
# default route
@app.route('/')
def index():
    return 'Hello World!!!!!!'

# function for responses
def results():
    # build a request object
    req = request.get_json(force=True)
    # fetch action from json

    verdict = "TRUTH"
    number_of_articles = 155

    article1 = "http://amulya.co"
    article2 = "http://amulya.co/article1"

    command = req.get('queryResult').get('parameters').get('Command')
    action = req.get('queryResult').get('intent').get('displayName')

    if action == "echo":
        stance = req.get('queryResult').get('parameters').get('echoText')
        sentence = "Your search was: '" + stance + "' "
        response = {
            "fulfillmentText": sentence,
            "source" : "TruthAI",  
        } 
        return response
         
    if action == "webhook-intent":
        stance = req.get('queryResult').get('parameters').get('stance')
      
        sentence = "Your search was: '" + stance + "' and your command was '" + command + "' | We referenced " + str(number_of_articles)  + " articles and our verdict about your stance is " + verdict + " | Here are a few more articles " + article1 + " and " + article2  + " for more info"

        try:
            stance
        except:
            sentence = "please try again"

        response = {
            "fulfillmentText": sentence,
            "source" : "TruthAI",  
            
            "payload": {
                "google": {
                "expectUserResponse": True,
                "richResponse": {
                    "items": [
                    {
                        "simpleResponse": {
                        "textToSpeech": "Howdy! I can tell you fun facts about almost any number."
                        }
                    },
                    {
                        "simpleResponse": {
                        "textToSpeech": "What number do you have in mind?"
                        }
                    }
                    ],
                    "suggestions": [
                    {
                        "title": "25"
                    },
                    {
                        "title": "45"
                    },
                    {
                        "title": "Never mind"
                    }
                    ],
                    "linkOutSuggestion": {
                    "destinationName": "Website",
                    "url": "https://assistant.google.com"
                    }
                }
                }
            }
        }
        # return a fulfillment response
        return response


# create a route for webhook

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    # return response
    
    
    return make_response(jsonify(results()))


# run the app
if __name__ == '__main__':
    app.run(debug=True)