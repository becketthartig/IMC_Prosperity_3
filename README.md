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

## Round 3
Round 3 brought Volcanic Rock and Volcanic Rock vouchers that acted like call options
for Volcanic Rock. Thinking about the many ways quant firms and options market-makers
trade options in real life, we look to finding mispricings in things like implied 
volatility and true volatility which was ultimately the strategy that we used to trade.
Using the Black-Scholes model, we were able to calculate options implied volatility and 
graph it against moneyness for all the historical data we had. The resulting graph 
pictured below, shows us that the implied volatility for all the available options often
deviated from a mean that we determined by fitting a 2-degree polynomial to the data.
As such we were able to profitably trade on mispriced volatility. Our strategy attempted,
to good success, making these trades and using delayed delta hedging, and an optimization
algorithm to determine the best security, option or underlying, to hedge delta with while 
taking into account other option mispricings and the best options to balance gamma. 

<img width="796" alt="PNG image" src="https://github.com/user-attachments/assets/adf2bbf6-9dec-4070-a7cc-915bdee62de1" />

## Round 4
This round really pushed Adi and I to expand our limits for personal progress. Bogged
down with Northwestern school work, and disappointed by our smaller progress in the
international leaderboard over the last round, this round was a challenge, but we 
knew we had to push through for glory, and honor the commitment of our late night
debugging session for the three previous rounds. 

Round 4 brought Magnificent Macarons, the opportunity to trade Magnificent Macarons
with a simulated nation abroad (including tariffs) and the sunlight index as well as 
sugar price. Ultimately we deciding on a trading strategy based on a "critical value"
for sunlight index to where low sunlight would spike the price of sugar, also leading
to an increase in the price of Magnificent Macarons, as well as the derivative of 
sunlight index. When sunlight was low and below the critical threshold, we would go 
short if derivative was positve or long if negative. Otherwise, we just
traded on mean reversion around a value of 622 Seashells (*Oh yeah, Seashells we the 
currency). While this worked in backtests, it proved to be overfit for Round 4, 
sadly getting killed in the simulation.

## Round 5
After getting killed in round 4, and having too much other work to do Adi and I were
more loose with the development in round 5. Round 5 brought de-anonymized trading, 
where you could look at patterns in when people trade that led you to find signals
on when to trade certain goods. While I'm sure that there were other signals to be
found, and I'd be interested to see what other people found in their writeups, we
found just one signal for Croissants relatively easily. Pictured below, you can see 
all the instances of two people trading Croissants over three days of data and we 
find that Caesar is an idiot and gets destroyed by Olivia in the trading game every 
day.

<img width="1191" alt="Screenshot 2025-04-27 at 11 38 17 PM" src="https://github.com/user-attachments/assets/25e1b92e-957d-4538-9900-3e503af6f9c8" />

## Finally
Thanks again to the team at IMC for putting on such a great event. Expect big things
to come from the Literal Zero for Prosperity 4!
