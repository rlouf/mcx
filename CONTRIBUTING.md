# Contributing to MCX

Thank you for opening this file and considering contributing to MCX! MCX is a library that is at its infancy, and there are many opportunities for anyone, experimented and less experimented developers, to makes important contributions. We consider as equally valuable:

- new features
- improvement in code quality and readability
- bug fixing
- improving the test suite
- documenting
- generally participating in a meaningful way to the community

## New features

We always welcome other people's ideas to implement new functionalities. There is only one thing that we will not accept:

* Variational inference

Not that we have anything against it, but one of this library's goal is to see how far we can go without it. Here are the things we are excited about:

* New distributions 
* New random layers
* Improvements to the compiler (new constructs, better management of the existing ones)
* Stochastic processes
* Times series
* New samplers. Specifically around managing discrete variables, sequential sampling and schemes that use the Hessian.

## Improving code quality

We try to live by the lessons of "Clean Code" or "Refactoring", but as busy humans, we sometimes fail. Any contribution, even just a name change in the internals, which makes the code easier to read and maintain is welcome. We think that nitpicking is a good thing.

## Improving the test suite

Adding new tests is great, and goes along with bug fixing. We also appreciate *reducing the number of useless tests* and generally making the test suite run faster. This can have a tremendous compounding effect; less time spent waiting for the tests to pass makes it easier to contribute.

## Fixing bugs

Start with writing a regression test that reproduces the bug in the simplest way. The bug is considered fixed when this new test passes along with the other ones.

## Documentation & Examples

Contributing to the documentation is a wonderful way to make a great impact to the library. Documentation is the first thing that new users see; you can help them avoid some of the confusions that you encountered when first using the library.

Examples are a great way for you to learn how to use the library and contribute in a meaningful way. We're happy to help anyone who contributes examples, so don't heistate to open a PR!

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
