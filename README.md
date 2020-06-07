# Vahvistusoppimista OpenAI Gym ympäristössä

Projektin tarkoituksena oli luoda oppiva tekoälyagentti johonkin ympäristöön. Algoritmiksi valitsin Q-learningin ja ympäristöksi OpenAI Gym Taxi-v3 ympäristön. Ympäristössä on tavoitteena viedä asiakas määränpäähän liikuttamalla taksia. Asiakkaan ja määränpään sijainti vaihtelevat satunnaisesti eri pelikeroilla. Projekti tehty osana yliopiston soveltava projekti kurssia.

*  Taksiympäristö Taxi-V3 https://gym.openai.com/envs/Taxi-v3/
*  Random ja Q-learning agentit
*  Palkkioiden ja rangaistuksien vertailu graafisesti

# Luokkien ja metodien kuvaus

## Agentit

/agents kansiossa on abstrakti yliluokka Agent, jolla kaksi metodia get_action ja get_policy
* get_action palauttaa toimen (ei välttämättä policyn mukainen)
* get_policy palauttaa policyn mukaisen toimen

Kansiossa on myös Q-learning algoritmia käyttävä agentti QLearningAgent sekä satunnaisia
toimia valitseva agentti RandomAgent. Nämä luokat perivät Agentin ja toteuttavat metodit.
QLearningAgent omaa myös metodin update, jolla päivitetään Q-taulun arvoja toiminnan jälkeen.

## Agentin treenaus ja evaluointi

main.py tiedostossa on kaksi metodia:

* training() metodi on agentin treenausta varten
* evaluate() metodi on agentin suorituksen arviointia varten

main.py tiedoston suoritus alustaa Gymin Taxi-v3 ympäristön, treenaa Q-learning
agentin sekä arvioi opetetun agentin sekä vertailun vuoksi satunnaisia toimia tekevän
agentin.

## Jupyter Notebook

Mukana olevassa notebook.ipynb tiedostossa on koodia erilaisten kaavioiden/diagrammien
luomista varten. Tämän saa avattua Jupyter Notebook nimisellä ohjelmalla.

# Ohjelman suoritus

```
pip install -r requirements.txt
python main.py
```

# Video

<a href="http://www.youtube.com/watch?feature=player_embedded&v=4miwRVk0tzs
" target="_blank"><img src="http://img.youtube.com/vi/4miwRVk0tzs/0.jpg" 
alt="Q learning video" width="240" height="180" border="10" /></a>
