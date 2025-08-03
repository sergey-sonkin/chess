# TODO

TODO:
* Improve

2025/08/01:
* Implement base model
* Improve file saving

## Notes on current reward function
 
How it works:
1. AI plays complete self-play games
2. At the end, we look at the game result (board.result())
3. Every position in that game gets labeled with the final outcome
4. If it was white's turn at that position, use the raw outcome
5. If it was black's turn, we flip the outcome (so black winning becomes +1.0 from black's perspective)

Limitations of this approach:
- Very sparse rewards - only at game end
- No positional understanding (material advantage, king safety, etc.)
- Positions early in the game get same reward as critical endgame positions
- No reward shaping for good moves vs bad moves within the same game

Potential improvements:
- Add material count rewards
- Reward check/checkmate threats
- Penalize unsafe king positions
- Use intermediate evaluations during training
- Weight positions differently based on game stage

