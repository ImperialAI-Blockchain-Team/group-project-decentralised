import web3 from './web3';
import {registry} from "../abi/abi";

const address = '0x84265ED9b257013ca77be6fFcCFd4c94ced054bE';

export default new web3.eth.Contract(registry, address);
