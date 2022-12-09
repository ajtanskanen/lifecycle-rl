# Life cycle model with reinforced learning & Python

A life cycle model describes agents making optimal decisions in the framework of 
Finnish social security. The main interest in this setting is how changes in the social security
impacts employment and public finances.

## Basis

The library depends on separate <a href='https://github.com/ajtanskanen/econogym'>econogym</a> and 
<a href='https://github.com/ajtanskanen/benefits'>benefits</a> modules. Econogym
implements how agents behave in the presence of detailed description of social security benefits and salary.
The optimal  behavior of the agents is solved using Reinforcement Learning library stable baselines 2, however,
other choices are also present.

## Results

The library reproduces the observed employment rates in Finland quite well, at all ages
from 18-70 separately for both women and men. 

<img src='https://github.com/ajtanskanen/lifecycle_rl/images/kuva1a.png'>

<img src='https://github.com/ajtanskanen/lifecycle_rl/images/kuva1a.png'>

<img src='https://github.com/ajtanskanen/lifecycle_rl/images/kuva1a.png'>

<img src='https://github.com/ajtanskanen/lifecycle_rl/images/kuva1a.png'>

<img src='https://github.com/ajtanskanen/lifecycle_rl/images/kuva1a.png'>

# How to run

Jupyter notebook or command line.

Clone library and install it by running command 

    pip install -e .

on the directory.

## References

	@misc{lifecycle_rl_,
	  author = {Antti J. Tanskanen},
	  title = {Elinkaarimalli},
	  year = {2019},
	  publisher = {GitHub},
	  journal = {GitHub repository},
	  howpublished = {\url{https://github.com/ajtanskanen/lifecycle_rl}},
	}

Description of the lifecycle model can be found from articles 
<a href='https://www.taloustieteellinenyhdistys.fi/wp-content/uploads/2020/06/KAK_2_2020_WEB-94-123.pdf'>Tanskanen (2020)</a> and 
<a href='https://www.sciencedirect.com/science/article/pii/S2590291122000171'>Tanskanen (2022)</a>.

    @article{tanskanen2022deep,
      title={Deep reinforced learning enables solving rich discrete-choice life cycle models to analyze social security reforms},
      author={Tanskanen, Antti J},
      journal={Social Sciences & Humanities Open},
      volume={5},
      pages={100263},
      year={2022}
    }
    
    @article{tanskanen2020tyollisyysvaikutuksien,
      title={Ty{\"o}llisyysvaikutuksien arviointia teko{\"a}lyll{\"a}: Unelmoivatko robotit ansiosidonnaisesta sosiaaliturvasta},
      author={Tanskanen, Antti J},
      journal={Kansantaloudellinen aikakauskirja},
      volume={2},
      pages={292--321},
      year={2020}
    }