import web3 from './web3';
import {registry} from "../abi/abi";

const address = '0x55Fd3A7eF427AA68E94759FbcCb4c75B490cE7e9';

export default new web3.eth.Contract(registry, address);