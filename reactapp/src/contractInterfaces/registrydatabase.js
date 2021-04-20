import web3 from './web3';
import {registry} from "../abi/abi";

const address = '0x55c866373ff96F386899c6da4729632e7D4c45f8';

export default new web3.eth.Contract(registry, address);
