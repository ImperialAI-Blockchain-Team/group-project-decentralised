// import Web3 from 'web3';

// const web3 = new Web3(window.web3.currentProvider);

// export default web3;

import Web3 from 'web3';

    var web3;

    if (typeof window) {
        if(!window.web3){
            const provider = new Web3.providers.HttpProvider(
                `https://ropsten.infura.io/v3/ec89decf66584cd984e5f89b6467f34f`
                );
            web3 = new Web3(provider);
        } else{
            web3 = new Web3(window.web3.currentProvider);
        }
    } else {
        const provider = new Web3.providers.HttpProvider(
            `https://ropsten.infura.io/v3/ec89decf66584cd984e5f89b6467f34f`
            );
        web3 = new Web3(provider);
    }


//const web3 = new Web3(window.web3.currentProvider);
export default web3;