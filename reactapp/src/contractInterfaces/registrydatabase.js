import web3 from './web3';
import {registry} from "../abi/abi";

const address = '0x9f7738115E97e1C564cecBC1d40176C35b94D0Cf';

export default new web3.eth.Contract(registry, address);
