# IMC_Prosperity_3
This repo holds the competition submission for team Literal Zero. 
Literal Zero finished #1 Costa Rica, #1 Central America and Caribbean, 
#3 Latin America.

First and foremost, thank you to IMC for hosting such a great [event](https://prosperity.imc.com/archipelago).
For the three weeks leading up to and during the event, my friend
Adi Rosenstock (from Costa Rica, hence the team nationality) and 
I worked incredibly hard on this complex, fast-paced competition and 
introduction to algorithmic trading. Please feel welcome to explore the code
or any tools used for our final submission, or read the write-up below.

## Round 1 and Tutorial:
The tutorial introduced the two tradeable resources, Rainforest Resin and Kelp. 
We quickly discovered Rainforest Resin acted like a quickly mean-reverting security
and, based on some research about IMC Prosperity 2, the previous years competition, 
discovered that Kelp could be traded profitably using a basic linear regression model
trained on the last 4 timesteps of the stock. We implemented a wide market-making spread
around 10000 for Rainforest Resin, and did a similar spread to market make around a 
linear regression predicted mid price for Kelp. After the tutorial, at the beginning of
round 1, a new commodity, Squid Ink, was introduced. 

