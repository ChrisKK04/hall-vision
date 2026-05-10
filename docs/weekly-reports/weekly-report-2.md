# Week 2 report

### Finnish

Luin tällä viikolla kirjan [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/) 1. kappaleen ja tein siinä olevat tehtävät. Implementoin myös kappaleen lopussa esitellyn ohjelman. Kokeilin myös neuroverkolle monia eri hyperparametreja, enkä saanut tulosta paranemaan käytännössä ollenkaan verrattuna kirjassa mainittuihin hyperparametreihin.

Ohjelma on edistynyt hyvin ja varsinainen neuroverkko on valmis. Lisäsin myös mahdollisuuden parametrien tallentamiseksi ja lataamiseksi.

Opin tällä viikolla kirjan 1. kappaleen asiat (perceptronit, sigmoid neuronit, yleistä neuroverkkojen rakennetta, MNISTIN tunnistavan neuroverkon rakenne + heuristiikkaa miksi toimii (vielä melko hämärää), virhefunktio, gradianttimenetelmä).

Vaikeinta oli gradianttimenetelmän ymmärtäminen. En myöskään tiedä yhtään miten käytetty takaisinvirtausalgoritmi toimii (toiminta on selitetty luvussa 2. minkä käyn läpi seuraavilla viikoilla). En vaikuta saavan yli 96 % tunnistus tarkkuutta (tavoitteena +97 %).

Seuraavaksi jatkan itse ohjelman koodaamista (ohjelma toimii nyt hyvin). Implementoin tavan omien lukujen tunnistukseen (tarvitsevat vain esikäsittelyn), visuaalisen tavan testidatan kuvien näkemiseksi + näillä testaaminen (käyttäjä näkee mitä kuvia on testidatassa ja voi valita niistä haluamansa) + tulosten näkeminen (mitkä oikein ja mitkä väärin visuaalisesti). Jatkan tämän jälkeen testaukseen ja siihen liittyviin asioihin. Jätän materiaalin 2. kappaleen (takaisinvirtausalgoritmin selitys) luultavasti seuraaville viikoille, sillä pyrin ensin saamaan itse ohjelman valmiiksi.

Käytin viikon aikana työhön noin 20 tuntia.

### English
This week I read the book's [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/) 1. chapter and did the associated exercises. I also implemented the program at the end of the chapter. I also tested many different hyperparameters and I couldn't get the result to improve at all compared to the hyperparameters mentioned in the book.

The program has progressed well and the neural network is ready. I also added functionality for parameter saving and loading.

This week I learnt the topics of the 1st chapter of the book (perceptrons, sigmoid neurons, general neural network structure, neural network structure for recognising MNIST + heuristics for why it works (still quite obscure), error function, gradient method).

Understanding the gradient method was the hardest part. I also don't understand how the used backpropagation algorithm works (it's explained in chapter 2, which I will go through in the following weeks). I can't seem to get above 96% recognition accuracy (the goal is +97%).

Next I will continue with the coding of the program (the program currently works well). I will implement a method for recognizing custom images (the images only need preprocessing), a visual way for seeing the images in the test data + testin with these (the user will be able to see what images are in the test data and select the desired ones) + seeing the results (what was correct and what was wrong visually). After this I will continue to testing and related matters. I will leave the materials 2nd chapter (the explanation of the backpropgation algorithm) probably for the following weeks, due to me wanting to finish the program first.

I worked on the project for around 20 hours.