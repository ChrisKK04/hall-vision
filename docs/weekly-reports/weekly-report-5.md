# Week 5 report

### Finnish

Toteutin [tämän](https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/) artikkelin mukaiset testit neuroverkolle (ylisovitus ja kaikki kerrokset muuttuvat + oikea verkon rakenteen alustus). Tein myös ensimmäisen veraisarvioinnin ja toteutin MNIST kuvien tulostuksen, mitä käytetään selvittämään millaisia kuvia verkko pystyy/ei pysty tunnistamaan.

Ohjelma edistyi hyvin.

Opin tällä viikolla lähinnä neuroverkkojen testaamiseen liittyviä seikkoja (+millaisia vaikeuksia neuroverkkojen testauksessa on). Sain myös oppitunnin pythonin osoittimista. Kun palautin verkon parametreja testausta varten, palautin oikeastaan vain tuplen, joissa oli osoittimia parametreihin, en varsinaisia parametreja itsessään. Täten jos "tallensin" verkon tilan ennen ja jälkeen muutosten, vanha tallennus oli sama kuin uusi, sillä tallensin alussa vain osoittimia, jotka osoittivat osoitteisiin, jotka päivitettiin. Tähän meni parin tunnin debuggaus. Opin myös lisää matplotlib:n käyttöä.

En ole varma ovatko testit riittävällä tasolla. Nyt testaan ylisovitus, kaikkien kerrosten toimivuus ja verkon oikean rakentaan alustus. En tiedä ovatko nämä riittäviä verkon testaamiseen. En ainakaan keksi enempää hyviä ja hyödyllisiä testejä.

Ohjelma on suurelta osin valmis. Täten jatkan testeistä jos niissä on puutteita ja pyrin myös parantamaan visualisointia. Tajusin myös että treenatessaan, verkko feedforwardaa vain yksi kuva kerrallaan. Jos aika jää, yritän nopeuttaa tätä toteuttamalla feedforwardauksen koko mini erälle kerralla (nyt syötteenä 784-vektori, koko mini erällä olisi 784-n-matriisi (n = mini erän kuvien määrä)) (en ole varma kuinka suuri nopeutus tällä saavutettaisiin).

Käytin viikon aikana työhön noin 15 tuntia.

### English

I implemented the neural network tests based on [this](https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/) article (overfitting and every layer changing + correct network structure initialization). I also did the first peer review and implememnted MNIST image printing, which is used for figuring out which types of images the network can/can't recognize.

The program progressed well.

This week I mainly learnt stuff related to neural network testing (+what types of challenges there are when it comes to neural network testing). I also got a lecture on Python pointers. When I was returning network parameters for testing, I was returning a tuple, that included pointers to the parameters, not the parameters themselves. Thus if I "saved" the network's state before and after changes, the old state was the same as the new one, because at the beginning I was saving pointers, that pointed to the addresses, that were updated. I spent a couple of hours debugging this. I also learnt more matplotlib usage.

I'm not sure if the tests are at an acceptable level. Currently I'm testing overfitting, if all layers work and correct network structure initialization. I'm not sure these are enough for testing the network. At least I can't come up with any good and useful tests.

The program is largely ready. Thus I will continue with the tests if they have shortcomings and I'll also try to improve the visualization. I also realized that when training, the network feedforwards one image at a time. If time is left over, I will try implement feedforwarding for an entire mini batch at once (now there's a 784-vector input, with the whole mini batch it would be a 784-n-matrix (n = the amount of images in the mini batch)) (I'm not sure how big of a speedup could be achieved with this).

I worked on the project for around 15 hours.