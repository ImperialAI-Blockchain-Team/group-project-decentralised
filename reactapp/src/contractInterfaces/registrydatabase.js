import web3 from './web3';
import {registry} from "../abi/abi";

const address = '0x2737C081A9E49a2f1D69fcc81474d3394f5916a1';

export default new web3.eth.Contract(registry, address);
