import web3 from './web3';
import {registry} from "../abi/abi";

const address = '0xE674Ac96DdAB6231D6C16bc0064e98c69F07b4B4';

export default new web3.eth.Contract(registry, address);
