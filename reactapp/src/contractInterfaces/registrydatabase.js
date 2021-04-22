import web3 from './web3';
import {registry} from "../abi/abi";

const address = '0xBfe725edDC643b7c200844842DeE972DAFa25712';

export default new web3.eth.Contract(registry, address);
