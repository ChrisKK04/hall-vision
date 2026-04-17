# Week 5 report

### Finnish

Toteutin [tämän](https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/) artikkelin mukaiset testit neuroverkolle (ylisovitus ja kaikki kerrokset muuttuvat + oikea verkon rakenteen alustus). Tein myös ensimmäisen veraisarvioinnin ja toteutin MNIST kuvien tulostuksen, mitä käytetään selvittämään millaisia kuvia verkko pystyy/ei pysty tunnistamaan.

Ohjelma edistyi hyvin.

Opin tällä viikolla lähinnä neuroverkkojen testaamiseen liittyviä seikkoja (+millaisia vaikeuksia neuroverkkojen testauksessa on). Sain myös oppitunnin pythonin osoittimista. Kun palautin verkon parametreja testausta varten, palautin oikeastaan vain tuplen, joissa oli osoittimia parametreihin, en varsinaisia parametreja itsessään. Täten jos "tallensin" verkon tilan ennen ja jälkeen muutosten, vanha tallennus oli sama kuin uusi, sillä tallensin alussa vain osoittimia, jotka osoittiviat osoitteisiin, jotka päivitettiin. Tähän meni parin tunnin debuggaus. Opin myös lisää matplotlib:n käyttöä.

En ole varma ovatko testit riittävällä tasolla. Nyt testaan ylisovitus, kaikkien kerrosten toimivuus ja verkon oikean rakentaan alustus. En tiedä ovatko nämä riittäviä verkon testaamiseen. En ainakaan keksi enempää hyviä ja hyödyllisiä testejä.

Ohjelma on suurelta osin valmis. Täten jatkan testeistä jos niissä on puutteita ja pyrin myös parantamaan visualisointia. Tajusin myös että treenatessaan, verkko feedforwardaa vain yksi kuva kerrallaan. Jos aika jää, yritän nopeuttaa tätä toteuttamalla feedforwardauksen koko mini erälle kerralla (nyt syötteenä 784-vektori, koko mini erällä olisi 784-n-matriisi (n = mini erän kuvien määrä)) (en ole varma kuinka suuri nopeutus tällä saavutettaisiin).

Käytin viikon aikana työhön noin 15 tuntia.

### English