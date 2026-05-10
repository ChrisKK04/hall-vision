# Week 3 report

### Finnish

Lisäsin tällä viikolla käyttäjän omien kuvien tunnistuksen (käyttäjä syöttää kuvan polun, oikean numeron ja taustan poistossa käytettävän kynnysarvon). Aloitin myös kirjan [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/) 2. kappaleen lukemista.

Ohjelma edistyi käyttäjien kuvien tunnistuksen lisäämisen osalta.

Opin tällä viikolla lähinnä kuvien esikäsittelyyn liittyviä asioita ja käsittelyyn tarvittavaa [OpenCV](https://opencv.org/) kirjaston käyttöä. Opin myös kirjan 2. kappaleen alun asioita (neuroverkon tulosteen nopea laskeminen matriiseilla).

En ole varma onko nykyinen esikäsittely paras mahdollinen. Nyt esikäsittely sisältää kuvien mustavalkoiseksi tekemisen, skaalauksen, taustamelun poiston, pikseli arvojen muuttamisen väliltä 0-255 välille 0-1, pikseli arvojen kääntämisen (musta -> valkoinen ja valkoinen -> musta) ja kuvan 784-vektoriksi kääntäminen. En tiedä pitäisikö kuvia käsitellä vielä jollain tavoin vai onko tämä riittävää.

Päätin seuraavaksi kirjoittaa network.py:ssä olevan neuroverkon täysin uusiksi itse. Nykyinen implementaatio on sama kuin kirjassa ja haluan yrittää toteuttaa erityisesti takaisinvirtausalgoritmin itse. Materiaalissa on hyvin tarkka kuvaus käytetystä [algoritmista](http://neuralnetworksanddeeplearning.com/chap2.html#the_backpropagation_algorithm), joten lähinnä vain käännän kuvauksen koodiksi.

Käytin viikon aikana työhön noin 10 tuntia.

### English

This week I added user provided image recognition (the user submits the path of the image, the correct digit and the threshold value used in background removal). 
I also started reading the book's [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/) 2nd chapter.

The program progressed when it comes to the user image recognition portion.

This week I mainly learnt topics on image preprocessing and the usage of the necessary [OpenCV](https://opencv.org/) libarary. I also learnt things from the beginning of the 2nd chapter of the book (fast computation of the output of a neural network using matrices).

I'm not sure if the current preprocessing is optimal. Currently, the preprocessing includes turning images into graysacle, resizing, background noise removal, updating pixel values from 0-255 to 0-1, inverting pixel values (black -> white and white -> black) and turning the image into a 784-vector. I'm not sure if the images should preprocessed further or if this is enough.

I've decided to fully rewrite the neural network in network.py. The current implementation is the same as in the book and I especially want to implement the backpropagation algorithm myself. There's a very detailed description of the used [algorithm](http://neuralnetworksanddeeplearning.com/chap2.html#the_backpropagation_algorithm) in the material, so I'll mainly be translating the description to code.

I worked on the project for around 10 hours.