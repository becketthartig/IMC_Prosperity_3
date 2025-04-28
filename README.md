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

## Round 1 and Tutorial
The Tutorial introduced the two tradeable resources, Rainforest Resin and Kelp. 
We quickly discovered Rainforest Resin acted like a quickly mean-reverting security
and, based on some research about IMC Prosperity 2, the previous years competition, 
discovered that Kelp could be traded profitably using a basic linear regression model
trained on the last 4 timesteps of the stock. We implemented a wide market-making spread
around 10000 for Rainforest Resin, and did a similar spread to market make around a 
linear regression predicted mid price for Kelp. After the tutorial, at the beginning of
Round 1, a new commodity, Squid Ink, was introduced. After about 2 long afternoons of 
analysis, building tools, and getting familiar with how to code for the trading environment, 
we figured, based on the hint from the dev team, that we could trade off a z-score with
mean 0. When the z-score jumped above a certain threshold, this was an indicator to us
that the stock was over/underpriced, and that it would quickly revert to the longer-term
mean.

## Round 2
Round 2 introduced the basic index funds Gift Basket 1 and Gift Basket 2, which both 
consisted of unique combinations of tradeable Croissants, Jams, and Djembes. Before
the three-day submission window for Round 2 ended, we weren't able to find a reliable
strategy for the underlying assets in the indices, Croissants, Jams, and Djembes, however
we did manage to find a reliable strategy in trading the index funds based on the 
synthetic basket values of their underlying. Assuming mean reversion again, as for Squid
Ink in Round 1, we again traded a z-score on the difference in price between gift baskets
and their synthetics. While this proved semi-profitable in the long run, given more time, 
we would've definitely liked to come up with a stronger model for this, as even if the 
z-score was high, that didn't always guarantee it was profitable to trade long/short, 
one way, every time.



