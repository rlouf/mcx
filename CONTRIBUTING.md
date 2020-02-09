# Contributing to MCX

Thank you for opening this file and considering contributing to MCX! MCX is a library that is at its infancy, and there are many opportunities for anyone, experimented and less experimented developers, to makes important contributions. We consider new features, improvement in code quality and readability, bug fixing, documenting and generally participating in a meaningful way to the community as equally valuable.

## Give back

MCX would not be possible without the great work put in other libraries such as Jax and Trax. Sometimes, imolementing new functionalities in MCX implies supplementing these libraries. Especially when adding new distributions: `jax.scipy.stats` implements relatively few distributions. It is important for us to give back to these libraires and push every feature in MCX that could be used by a wider audience:

- Many users of these libraries probably need these features; Python is not only a language, but and ecosystem. We need to not only tend to our library but the ecosystem as a whole;
- A utility that lives in these libraires will probably get more attention than here; this improves code quality;
- This is just the right thing to do and show these libraries' author we are grateful for their work and willing to give back.

As a result, the process to contribute a new distribution is the following:

1. Open a PR on mcx with all tests passing;
2. Go through the usual code review;
3. Once your code is ready to merge, open a PR on JAX;
4. Merge your original PR on mcx;
5. Once the changes have been accepted on JAX and released, import the functionality directly from JAX.

Any distribution implementation on JAX is considered as a contribution to this project as well.
