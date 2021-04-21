import web3 from './web3';
import {registry} from "../abi/abi";

const address = '0x3b980b068997E4be04fE19DefC1622D9bA7da8Ee';

export default new web3.eth.Contract(registry, address);
