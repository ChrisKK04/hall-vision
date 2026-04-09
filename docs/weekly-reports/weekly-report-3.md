# Week 3 report

### Finnish

Lisäsin tällä viikolla käyttäjän omien kuvien tunnistuksen (käyttäjä syöttää kuvan polun, oikean numeron ja taustan poistossa käytettävän kynnysarvon). Aloitin myös kirjan [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/) 2. kappaleen lukemista.

Ohjelma edistyi käyttäjien kuvien tunnistuksen lisäämisen osalta.

Opin tällä viikolla lähinnä kuvien esikäsittelyyn liittyviä asioita ja käsittelyyn tarvittavaa [OpenCV](https://opencv.org/) kirjaston käyttöä. Opin myös kirjan 2. kappaleen alun asioita (neuroverkon tulosteen nopea laskeminen matriiseilla).

En ole varma onko nykyinen esikäsittely paras mahdollinen. Nyt esikäsittely sisältää kuvien mustavalkoiseksi tekemisen, skaalauksen, taustamelun poiston, pikseli arvojen muuttamisen väliltä 0-255 välille 0-1, pikseli arvojen kääntämisen (musta -> valkoinen ja valkoinen -> musta) ja kuvan 784-vektoriksi kääntäminen. En tiedä pitäisikö kuvia käsitellä vielä jollain tavoin vai onko tämä riittävää.

Päätin seuraavaksi kirjoittaa network.py:ssä olevan neuroverkon täysin uusiksi itse. Nykyinen implementaatio on sama kuin kirjassa ja haluan yrittää toteuttaa erityisesti takaisinvirtausalgoritmin itse. Materiaalissa on hyvin tarkka kuvaus käytetystä [algoritmista](http://neuralnetworksanddeeplearning.com/chap2.html#the_backpropagation_algorithm), joten lähinnä vain käännän kuvauksen koodiksi.

Käytin viikon aikana työhön noin 10 tuntia.

### English